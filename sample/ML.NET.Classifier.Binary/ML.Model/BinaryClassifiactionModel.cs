using Microsoft.ML;
using Microsoft.ML.Data;
using ML.DataPreparation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Model
{
    /// <summary>
    /// a model for binary classification that supports two algorithms:
    /// - logistic regression (with l1/l2 regularization and support for class weighting)
    /// - averaged perceptron (faster, but doesn't output probabilities)
    /// 
    /// this class handles training, evaluation, threshold tuning (based on f1-score),
    /// and extracting predictions (score or probability) for charting or analysis
    /// </summary>
    public class BinaryClassificationModel : IBinaryClassificationModel
    {
        private readonly MLContext _ml;
        private ITransformer _model;               // trained ml.net model
        private double _threshold = 0.5;           // default classification threshold
        private bool _hasProbability = false;      // whether the model outputs calibrated probabilities

        private const string LabelCol = "Label";           // label column name
        private const string FeatCol = "Features";         // feature vector column name
        private const string PredCol = "PredictedLabel";   // output predicted label
        private const string ScoreCol = "Score";           // raw model output 
        private const string ProbCol = "Probability";      // predicted probability (if supported)

        /// <summary>
        /// creates an instance of the binary classifier using a shared mlcontext
        /// </summary>
        public BinaryClassificationModel(MLContext mlContext)
        {
            _ml = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        }

        /// <summary>
        /// trains the logistic regression model with l1/l2 regularization and weighting support
        /// </summary>
        public ITransformer TrainLogisticRegression(IDataView train)
        {
            RequireBothClasses(train, "Training");

            // automatically compute instance weights for class imbalance
            var weightedTrain = WeightingHelper.CreateWeightEstimator(_ml, train)
                                               .Fit(train)
                                               .Transform(train);

            var opts = new Microsoft.ML.Trainers.LbfgsLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = LabelCol,             // binary label column
                FeatureColumnName = FeatCol,            // input feature vector
                ExampleWeightColumnName = "Weight",     // optional weight column to handle imbalance
                L1Regularization = 0.01f,               // promotes sparse models (set some weights to 0)
                L2Regularization = 0.05f,               // avoids large weights, improves generalization
                MaximumNumberOfIterations = 150         // max optimization steps
            };

            var pipeline = _ml.Transforms.NormalizeMeanVariance(FeatCol)
                .Append(_ml.BinaryClassification.Trainers.LbfgsLogisticRegression(opts));

            _model = pipeline.Fit(weightedTrain);
            _hasProbability = true; // logistic regression supports probability output
            _threshold = 0.5; // reset after every successful training

            return _model;
        }

        /// <summary>
        /// trains the averaged perceptron model
        /// </summary>
        public ITransformer TrainAveragedPerceptron(IDataView train)
        {
            RequireBothClasses(train, "Training");

            var opts = new Microsoft.ML.Trainers.AveragedPerceptronTrainer.Options
            {
                LabelColumnName = LabelCol,     // label column
                FeatureColumnName = FeatCol,    // feature column
                NumberOfIterations = 20         // how many passes over the data
            };

            var pipeline = _ml.Transforms.NormalizeMeanVariance(FeatCol)
                .Append(_ml.BinaryClassification.Trainers.AveragedPerceptron(opts));

            _model = pipeline.Fit(train);
            _hasProbability = false; // perceptron does not produce probability
            _threshold = 0.0; // raw score decision boundary

            return _model;
        }

        /// <summary>
        /// returns ML.NET default threshold metrics for existing callers
        /// use EvaluateThresholdMetrics for classification metrics at the tuned threshold
        /// </summary>
        public BinaryClassificationMetrics EvaluateModel(IDataView test)
        {
            EnsureTrained();
            RequireBothClasses(test, "Evaluation");
            var predictions = Transform(test);

            return _hasProbability
                ? _ml.BinaryClassification.Evaluate(predictions,
                    labelColumnName: LabelCol, scoreColumnName: ScoreCol,
                    probabilityColumnName: ProbCol, predictedLabelColumnName: PredCol)
                : _ml.BinaryClassification.EvaluateNonCalibrated(predictions,
                    labelColumnName: LabelCol, scoreColumnName: ScoreCol,
                    predictedLabelColumnName: PredCol);
        }

        /// <summary>
        /// evaluates one cached prediction set and counts the actual decision labels
        /// </summary>
        public BinaryEvaluationResult EvaluateThresholdMetrics(IDataView test)
        {
            EnsureTrained();
            RequireBothClasses(test, "Evaluation");
            var predictions = _ml.Data.Cache(Transform(test));
            var rows = _ml.Data.CreateEnumerable<BinaryEvaluationRow>(
                predictions, reuseRowObject: false).ToList();
            BinaryClassificationMetrics defaultMetrics = _hasProbability
                ? _ml.BinaryClassification.Evaluate(predictions,
                    labelColumnName: LabelCol, scoreColumnName: ScoreCol,
                    probabilityColumnName: ProbCol, predictedLabelColumnName: PredCol)
                : _ml.BinaryClassification.EvaluateNonCalibrated(predictions,
                    labelColumnName: LabelCol, scoreColumnName: ScoreCol,
                    predictedLabelColumnName: PredCol);
            return new BinaryEvaluationResult(defaultMetrics, rows, _threshold, _hasProbability);
        }

        /// <summary>
        /// applies training-fitted preprocessing and the current decision threshold
        /// input needs Features, Label is not required for inference
        /// </summary>
        public IDataView Transform(IDataView data)
        {
            EnsureTrained();
            if (data == null) throw new ArgumentNullException(nameof(data));
            var predictions = _model.Transform(data);
            // IDataView is lazy: capture this threshold so later tuning cannot change this view
            double threshold = _threshold;
            if (_hasProbability)
            {
                var mapping = _ml.Transforms.CustomMapping<ProbabilityInput, PredOut>(
                    (src, dst) => dst.PredictedLabel = src.Probability >= threshold, null);
                return mapping.Fit(predictions).Transform(predictions);
            }
            else
            {
                var mapping = _ml.Transforms.CustomMapping<ScoreInput, PredOut>(
                    (src, dst) => dst.PredictedLabel = src.Score >= threshold, null);
                return mapping.Fit(predictions).Transform(predictions);
            }
        }

        /// <summary>
        /// extracts raw scores and true labels for ROC visualization for either algorithm
        /// </summary>
        public (List<float> probs, List<bool> labels) GetPredictionsAndLabels(IDataView testData)
        {
            EnsureTrained();
            if (testData == null) throw new ArgumentNullException(nameof(testData));
            var predictions = _model.Transform(testData);

            // Raw scores keep the graph's ranking input identical to ML.NET AUC
            // The tuple name "probs" is retained for source compatibility
            var rows = _ml.Data.CreateEnumerable<PredWithScore>(predictions, reuseRowObject: false).ToList();
            return (rows.Select(r => r.Score).ToList(), rows.Select(r => r.Label).ToList());
        }

        /// <summary>
        /// finds the threshold that maximizes f1 score on validation data
        /// </summary>
        public double OptimizeThresholdByF1(IDataView validationData)
        {
            EnsureTrained();
            _threshold = BinaryThresholdOptimizer.OptimizeThresholdByF1(
                model: _model,
                _ml,
                validationData,
                FeatCol,
                _hasProbability,
                out _ // ignore f1 value
            );
            return _threshold;
        }

        private void EnsureTrained()
        {
            if (_model == null)
                throw new InvalidOperationException("Train the model before predicting or tuning.");
        }

        private void RequireBothClasses(IDataView data, string stage)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            bool positive = false, negative = false;
            foreach (var row in _ml.Data.CreateEnumerable<LabelInput>(data, reuseRowObject: false))
            {
                if (row.Label) positive = true;
                else negative = true;
                if (positive && negative) return;
            }
            throw new ArgumentException(stage +
                " data must contain both positive and negative labels. Check the target column and split.",
                nameof(data));
        }

        private sealed class LabelInput
        {
            public bool Label { get; set; }
        }

        private sealed class ProbabilityInput
        {
            public float Probability { get; set; }
        }

        private sealed class ScoreInput
        {
            public float Score { get; set; }
        }

        // helper class for models that return both score and probability
        private sealed class PredWithProb
        {
            public float Probability { get; set; }   // predicted probability
            public float Score { get; set; }         // raw model score
            public bool Label { get; set; }          // ground truth label
        }

        // helper class for models that return only score (like perceptron)
        private sealed class PredWithScore
        {
            public float Score { get; set; }         // raw score only
            public bool Label { get; set; }          // ground truth label
        }

        // used to output final predicted label after applying threshold
        private sealed class PredOut
        {
            public bool PredictedLabel { get; set; } // final decision based on threshold
        }
    }
}
