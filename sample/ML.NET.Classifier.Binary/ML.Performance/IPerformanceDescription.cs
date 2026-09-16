using System.Collections.Generic;
using System.Drawing;

public interface IPerformanceDescription
{
    string GetReceiverCurveDescription();

    List<(string text, Color color)> GetMetricsReceiverCurve(
        double areaUnderRocCurve);
}