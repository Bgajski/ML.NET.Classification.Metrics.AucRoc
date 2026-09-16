using System.Collections.Generic;
using System.Windows.Forms.DataVisualization.Charting;

public interface IPerformanceVisualizer
{
    double VisualizeReceiverCurve(
        IEnumerable<float> scores,
        IEnumerable<bool> labels,
        Chart chart,
        string modelName);
}