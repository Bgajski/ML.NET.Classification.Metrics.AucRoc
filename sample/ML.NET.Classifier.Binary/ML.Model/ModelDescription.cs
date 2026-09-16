using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace ML.Model
{
    public class ModelDescription : IModelDescription
    {
        public string GetDescription(string algorithmName)
        {
            return algorithmName switch
            {};
        }

        public List<(string text, Color color)> GetMetricsDescriptionBinary(BinaryEvaluationResult metrics)
        {
            if (metrics == null) throw new ArgumentNullException(nameof(metrics));
            return new List<(string text, Color color)>
            {
                ("Accuracy: ", Color.Red), ($"{metrics.Accuracy:F2}", Color.Green),
                ("Precision: ", Color.Red), ($"{metrics.PositivePrecision:F2}", Color.Green),
                ("Recall: ", Color.Red), ($"{metrics.PositiveRecall:F2}", Color.Green)
            };
        }

        public List<(string text, Color color)> GetMetricsDescriptionBinary(BinaryClassificationMetrics metrics)
        {
            if (metrics == null) throw new ArgumentNullException(nameof(metrics));
            return new List<(string text, Color color)>
            {
                ("Accuracy: ", Color.Red), ($"{metrics.Accuracy:F2}", Color.Green),
                ("Precision: ", Color.Red), ($"{metrics.PositivePrecision:F2}", Color.Green),
                ("Recall: ", Color.Red), ($"{metrics.PositiveRecall:F2}", Color.Green)
            };
        }
    }
}
