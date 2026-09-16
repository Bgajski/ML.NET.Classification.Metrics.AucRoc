using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML.DataPreparation
{
    /// <summary>
    /// prepares binary classification datasets and applies optional instance weighting
    /// used for models like logistic regression and averaged perceptron
    /// </summary>
    public class BinaryDataPreparation : IBinaryDataPreparation, IDataPreparation
    {
        public (IDataView TrainingData, IDataView TestData)
            PrepareData(MLContext ml, string path, int featureCount) =>
            Prepare(ml, path);

        public (IDataView TrainingData, IDataView TestData)
            PrepareDataForLogisticRegression(MLContext ml, string path, int featureCount) =>
            Prepare(ml, path, useWeighting: true); // use instance weighting for logistic regression

        public (IDataView TrainingData, IDataView TestData)
            PrepareDataForAveragedPerceptron(MLContext ml, string path, int featureCount) =>
            Prepare(ml, path, useWeighting: false); // no weighting for averaged perceptron

        /// <summary>
        /// internal preparation pipeline for binary data
        /// handles label detection, column parsing, class-wise splitting, and optional weighting
        /// </summary>
        private static (IDataView TrainingData, IDataView TestData) Prepare(MLContext ml, string csvPath, bool useWeighting = false)
        {
            // detect label column index based on common names or binary patterns
            string[] header = File.ReadLines(csvPath).First().Split(',')
                .Select(name => name.Trim().Trim('\"').Trim('\uFEFF')).ToArray();
            int labelIdx = DetectLabelIndex(csvPath, header);
            if (labelIdx < 0) throw new Exception("No binary label column found.");

            // define columns for TextLoader
            var columns = new List<TextLoader.Column>
            {
                new("Label", DataKind.Boolean, labelIdx)
            };

            // create feature columns with names F0, F1, ...
            int featOrdinal = 0;
            for (int i = 0; i < header.Length; i++)
            {
                if (i == labelIdx) continue;
                columns.Add(new($"F{featOrdinal++}", DataKind.Single, i));
            }

            // load raw CSV using configured schema
            var loader = ml.Data.CreateTextLoader(new TextLoader.Options
            {
                Columns = columns.ToArray(),
                HasHeader = true,
                Separators = new[] { ',' }
            });

            var raw = loader.Load(csvPath);

            string[] featCols = columns.Where(c => c.Name.StartsWith("F")).Select(c => c.Name).ToArray();
            if (featCols.Length == 0)
                throw new ArgumentException("The dataset must contain at least one feature column.");

            var featureData = ml.Transforms.Concatenate("Features", featCols)
                .Fit(raw).Transform(raw);
            // Keep enough training examples for the form's train/validation/test split
            var split = SplitBinaryData(ml, featureData, 0.2, 42,
                "Loaded column '" + header[labelIdx] + "'", minimumTrainPerClass: 3);

            // Return unscaled features. The model fits normalization on its final
            // training partition, after the form has separated validation data
            var transformedTrain = split.TrainSet;
            var transformedTest = split.TestSet;

            // apply instance weighting if requested
            if (useWeighting)
            {
                var weightedTrain = WeightingHelper.CreateWeightEstimator(ml, transformedTrain)
                                                   .Fit(transformedTrain)
                                                   .Transform(transformedTrain);
                return (weightedTrain, transformedTest);
            }
            else
            {
                return (transformedTrain, transformedTest);
            }
        }

        /// <summary>
        /// tries to locate a binary label column using header keywords or value patterns
        /// </summary>
        private static int DetectLabelIndex(string path, string[] header)
        {
            var common = new[] { "label", "outcome", "class", "target", "y" };
            for (int i = 0; i < header.Length; i++)
                if (common.Contains(header[i], StringComparer.OrdinalIgnoreCase))
                    return i;

            // fallback: scan first 200 rows for binary values in each column
            var sample = File.ReadLines(path).Skip(1).Take(200).Select(l => l.Split(',')).ToArray();
            for (int col = 0; col < header.Length; col++)
            {
                var distinct = new HashSet<string>();
                foreach (var row in sample)
                {
                    if (col >= row.Length) continue;
                    var v = row[col].Trim().ToLowerInvariant();
                    if (!string.IsNullOrWhiteSpace(v)) distinct.Add(v);
                    if (distinct.Count > 2) break;
                }

                // check if values are binary compatible
                if (distinct.Count == 2 && distinct.All(IsBinary)) return col;
            }

            return -1;
        }

        // check if string represents a binary value
        private static bool IsBinary(string v) =>
            v is "0" or "1" or "false" or "true";

        // generate generic feature names: F0, F1, F2...
        public static string[] GenerateFeatureColumnNames(int n) =>
            Enumerable.Range(0, n).Select(i => $"F{i}").ToArray();

        // Splits the Label/Features data used by this application's binary training flow
        // Each row belongs to exactly one output; neither output can lose a class
        public static (IDataView TrainSet, IDataView TestSet) SplitBinaryData(
            MLContext ml, IDataView data, double testFraction, int seed, string stage,
            int minimumTrainPerClass = 1, int minimumTestPerClass = 1)
        {
            if (ml == null) throw new ArgumentNullException(nameof(ml));
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (testFraction <= 0 || testFraction >= 1 || double.IsNaN(testFraction))
                throw new ArgumentOutOfRangeException(nameof(testFraction));
            if (minimumTrainPerClass < 1 || minimumTestPerClass < 1)
                throw new ArgumentException("Each partition must retain at least one row per class.");

            var rows = ml.Data.CreateEnumerable<FeatureRow>(data, reuseRowObject: false).ToList();
            var positives = rows.Where(row => row.Label).ToList();
            var negatives = rows.Where(row => !row.Label).ToList();
            int required = minimumTrainPerClass + minimumTestPerClass;
            if (positives.Count < required || negatives.Count < required)
                throw new ArgumentException(
                    $"{stage}: {rows.Count} rows; positive labels: {positives.Count}; " +
                    $"negative labels: {negatives.Count}. At least {required} rows of each class " +
                    "are needed for this split. Check the target column and the CSV data.");

            var random = new Random(seed);
            var trainRows = new List<FeatureRow>();
            var testRows = new List<FeatureRow>();
            AddClassSplit(positives, trainRows, testRows, random, testFraction,
                minimumTrainPerClass, minimumTestPerClass);
            AddClassSplit(negatives, trainRows, testRows, random, testFraction,
                minimumTrainPerClass, minimumTestPerClass);
            Shuffle(trainRows, random);
            Shuffle(testRows, random);

            var featureType = data.Schema["Features"].Type as VectorDataViewType;
            if (featureType == null || featureType.Size == 0)
                throw new ArgumentException("Features must be a vector with a known size.");
            var schema = SchemaDefinition.Create(typeof(FeatureRow));
            schema[nameof(FeatureRow.Features)].ColumnType = featureType;
            return (ml.Data.LoadFromEnumerable(trainRows, schema),
                    ml.Data.LoadFromEnumerable(testRows, schema));
        }

        private static void AddClassSplit(List<FeatureRow> rows, List<FeatureRow> train,
            List<FeatureRow> test, Random random, double fraction, int minTrain, int minTest)
        {
            Shuffle(rows, random);
            int testCount = Math.Max(minTest, Math.Min(rows.Count - minTrain,
                (int)Math.Round(rows.Count * fraction, MidpointRounding.AwayFromZero)));
            test.AddRange(rows.Take(testCount));
            train.AddRange(rows.Skip(testCount));
        }

        private static void Shuffle(List<FeatureRow> rows, Random random)
        {
            for (int i = rows.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                var row = rows[i];
                rows[i] = rows[j];
                rows[j] = row;
            }
        }

        private sealed class FeatureRow
        {
            public bool Label { get; set; }
            public float[] Features { get; set; } = Array.Empty<float>();
        }
    }
}
