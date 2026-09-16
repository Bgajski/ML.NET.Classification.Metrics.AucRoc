using Microsoft.ML;
using System;
using System.Linq;

namespace ML.Model
{
    /// <summary>
    /// finds the best F1 decision threshold using held out validation data
    /// supports probabilities and unrestricted raw scores
    /// </summary>
    public static class BinaryThresholdOptimizer
    {
        // featureColumn is retained for compatibility with existing callers
        // The supplied model must include its training fitted preprocessing
        public static double OptimizeThresholdByF1(
            ITransformer model,
            MLContext mlContext,
            IDataView validationData,
            string featureColumn,
            bool hasProbability,
            out double bestF1)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (mlContext == null) throw new ArgumentNullException(nameof(mlContext));
            if (validationData == null) throw new ArgumentNullException(nameof(validationData));

            var predictions = model.Transform(validationData);
            var rows = hasProbability
                ? mlContext.Data.CreateEnumerable<PredWithProb>(predictions, reuseRowObject: false)
                    .Select(r => (Value: r.Probability, Label: r.Label)).ToList()
                : mlContext.Data.CreateEnumerable<PredWithScore>(predictions, reuseRowObject: false)
                    .Select(r => (Value: r.Score, Label: r.Label)).ToList();

            if (rows.Count == 0)
                throw new ArgumentException("Validation data is empty.", nameof(validationData));
            if (rows.Any(r => float.IsNaN(r.Value) || float.IsInfinity(r.Value)))
                throw new ArgumentException("Validation predictions must be finite.", nameof(validationData));
            if (hasProbability && rows.Any(r => r.Value < 0 || r.Value > 1))
                throw new ArgumentException("Probabilities must be between zero and one.", nameof(validationData));

            int positives = rows.Count(r => r.Label);
            if (positives == 0 || positives == rows.Count)
                throw new ArgumentException(
                    "Validation data must contain both positive and negative labels. Use a stratified split.",
                    nameof(validationData));

            // Include the default threshold and retain it if it already maximizes F1
            double defaultThreshold = hasProbability ? 0.5 : 0.0;
            double bestThreshold = defaultThreshold;
            int defaultTp = rows.Count(r => r.Value >= defaultThreshold && r.Label);
            int defaultFp = rows.Count(r => r.Value >= defaultThreshold && !r.Label);
            bestF1 = F1(defaultTp, defaultFp, positives - defaultTp);

            // Each distinct score defines a possible decision set with >= threshold
            // Sweep tied scores together
            // Sorting costs O(n log n), counting is O(n)
            int tp = 0, fp = 0;
            foreach (var group in rows.GroupBy(r => r.Value).OrderByDescending(g => g.Key))
            {
                foreach (var row in group)
                {
                    if (row.Label) tp++;
                    else fp++;
                }

                double candidateF1 = F1(tp, fp, positives - tp);
                double candidateThreshold = group.Key;
                if (candidateF1 > bestF1 ||
                    (candidateF1 == bestF1 &&
                     Math.Abs(candidateThreshold - defaultThreshold) <
                     Math.Abs(bestThreshold - defaultThreshold)))
                {
                    bestF1 = candidateF1;
                    bestThreshold = candidateThreshold;
                }
            }
            // Predicting no positives has F1 = 0, so it cannot beat this sweep
            // when validation contains at least one positive label
            return bestThreshold;
        }

        private static double F1(int tp, int fp, int fn)
        {
            double denominator = 2.0 * tp + fp + fn;
            return denominator > 0 ? 2.0 * tp / denominator : 0;
        }

        private sealed class PredWithProb
        {
            public float Probability { get; set; }
            public bool Label { get; set; }
        }

        private sealed class PredWithScore
        {
            public float Score { get; set; }
            public bool Label { get; set; }
        }
    }
}
