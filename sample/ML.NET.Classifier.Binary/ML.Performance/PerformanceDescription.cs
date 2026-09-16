using System;
using System.Collections.Generic;
using System.Drawing;

namespace ML.Performance
{
    public class PerformanceDescription : IPerformanceDescription
    {
        public string GetReceiverCurveDescription()
        {
            return "";
        }

        public List<(string text, Color color)>
            GetMetricsReceiverCurve(double areaUnderRocCurve)
        {
            if (!double.IsFinite(areaUnderRocCurve) ||
                areaUnderRocCurve < 0.0 ||
                areaUnderRocCurve > 1.0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(areaUnderRocCurve),
                    "AUC must be a finite value between zero and one.");
            }

            return new List<(string text, Color color)>
            {
                (
                    "Area Under the ROC Curve (AUC): ",
                    Color.Red
                ),
                (
                    $"{areaUnderRocCurve:F2}",
                    Color.Green
                )
            };
        }
    }
}