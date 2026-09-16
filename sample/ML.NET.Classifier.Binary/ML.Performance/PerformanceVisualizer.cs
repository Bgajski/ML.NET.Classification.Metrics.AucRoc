using System.Collections.Generic;
using System.Windows.Forms.DataVisualization.Charting;
using ML.Performance;

public class PerformanceVisualizer : IPerformanceVisualizer
{
    public double VisualizeReceiverCurve(
        IEnumerable<float> scores,
        IEnumerable<bool> labels,
        Chart chart,
        string modelName)
    {
        return ReceiverCurveCharacteristic.ShowReceiverCurve(
            scores,
            labels,
            chart,
            modelName);
    }
}