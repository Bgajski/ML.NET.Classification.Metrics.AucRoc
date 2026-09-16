using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Model
{
    /// <summary>
    /// Classification metrics counted from the actual
    /// threshold adjusted decisions
    ///
    /// ROC and AUC use the original raw scores,
    /// independently of that threshold
    /// </summary>
    public sealed class BinaryEvaluationResult
    {
        public int TruePositives { get; }

        public int FalsePositives { get; }

        public int TrueNegatives { get; }

        public int FalseNegatives { get; }

        public int ExampleCount =>
            TruePositives +
            FalsePositives +
            TrueNegatives +
            FalseNegatives;

        public double Threshold { get; }

        public bool UsesProbabilityThreshold { get; }

        public double Accuracy =>
            (double)(TruePositives + TrueNegatives) /
            ExampleCount;

        // Define precision as zero when no examples
        // are predicted positive
        public double PositivePrecision =>
            Divide(
                TruePositives,
                TruePositives + FalsePositives);

        public double PositiveRecall =>
            Divide(
                TruePositives,
                TruePositives + FalseNegatives);

        public double F1Score =>
            Divide(
                2.0 * TruePositives,
                2.0 * TruePositives +
                FalsePositives +
                FalseNegatives);

        // ML.NET AUC is retained as an independent reference value
        // The AUC shown in the application is returned by
        // ML.NET.Classification.Metrics.AucRoc
        public double AreaUnderRocCurve =>
            DefaultThresholdMetrics.AreaUnderRocCurve;

        public IReadOnlyList<float> Scores { get; }

        public IReadOnlyList<bool> Labels { get; }

        // Retained for ML.NET AUC/probability metrics only
        // Its classification metrics use ML.NET's default score
        // boundary, not tuned threshold
        public BinaryClassificationMetrics
            DefaultThresholdMetrics
        { get; }

        internal BinaryEvaluationResult(
            BinaryClassificationMetrics defaultMetrics,
            List<BinaryEvaluationRow> rows,
            double threshold,
            bool usesProbabilityThreshold)
        {
            DefaultThresholdMetrics =
                defaultMetrics ??
                throw new ArgumentNullException(
                    nameof(defaultMetrics));

            if (rows == null)
            {
                throw new ArgumentNullException(
                    nameof(rows));
            }

            if (rows.Count == 0 ||
                !rows.Any(row => row.Label) ||
                !rows.Any(row => !row.Label))
            {
                throw new ArgumentException(
                    "Evaluation needs both positive and negative examples.",
                    nameof(rows));
            }

            if (rows.Any(row =>
                    float.IsNaN(row.Score) ||
                    float.IsInfinity(row.Score)))
            {
                throw new ArgumentException(
                    "Evaluation scores must be finite.",
                    nameof(rows));
            }

            Threshold = threshold;
            UsesProbabilityThreshold =
                usesProbabilityThreshold;

            TruePositives = rows.Count(row =>
                row.Label &&
                row.PredictedLabel);

            FalsePositives = rows.Count(row =>
                !row.Label &&
                row.PredictedLabel);

            TrueNegatives = rows.Count(row =>
                !row.Label &&
                !row.PredictedLabel);

            FalseNegatives = rows.Count(row =>
                row.Label &&
                !row.PredictedLabel);

            // These two collections remain positionally aligned
            Scores = rows
                .Select(row => row.Score)
                .ToList()
                .AsReadOnly();

            Labels = rows
                .Select(row => row.Label)
                .ToList()
                .AsReadOnly();
        }

        private static double Divide(
            double numerator,
            double denominator)
        {
            return denominator > 0
                ? numerator / denominator
                : 0;
        }
    }

    internal sealed class BinaryEvaluationRow
    {
        public float Score { get; set; }

        public bool Label { get; set; }

        public bool PredictedLabel { get; set; }
    }
}