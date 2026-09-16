using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Drawing;

namespace ML.Model
{
    public interface IModelDescription
    {
        List<(string text, Color color)> GetMetricsDescriptionBinary(BinaryEvaluationResult metrics);
        string GetDescription(string algorithmName);
        List<(string text, Color color)> GetMetricsDescriptionBinary(BinaryClassificationMetrics metrics);
    }
}
