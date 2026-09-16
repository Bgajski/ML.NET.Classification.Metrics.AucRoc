using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms.DataVisualization.Charting;
using ML.NET.Metrics.Evaluation.Public;

namespace ML.Performance
{
    /// <summary>
    /// Displays a Receiver Operating Characteristic curve for a binary classifier
    /// The curve points and AUC come from the same native C++ calculation
    /// </summary>
    public class ReceiverCurveCharacteristic
    {
        private static readonly List<Color> ColorPalette = new()
        {
            Color.FromArgb(52, 152, 219),
            Color.FromArgb(231, 76, 60),
            Color.FromArgb(46, 204, 113)
        };

        public static double ShowReceiverCurve(
            IEnumerable<float> scores,
            IEnumerable<bool> labels,
            Chart chart,
            string modelName)
        {
            ArgumentNullException.ThrowIfNull(chart);

            // Calculate and validate before clearing an existing graph
            MetricsCurve<RocPoint> curve =
                CalculateReceiverCurve(scores, labels);

            chart.Series.Clear();
            chart.ChartAreas.Clear();
            chart.Legends.Clear();

            var chartArea = new ChartArea
            {
                AxisX =
                {
                    Title = "False Positive Rate",
                    TitleFont = new Font("Tahoma", 10)
                },
                AxisY =
                {
                    Title = "True Positive Rate",
                    TitleFont = new Font("Tahoma", 10)
                }
            };

            chart.ChartAreas.Add(chartArea);

            var rocSeries = new Series
            {
                Name = "ROC Curve",
                ChartType = SeriesChartType.Line,
                Color = ColorPalette[0],
                BorderWidth = 2,
                XValueType = ChartValueType.Double,
                YValueType = ChartValueType.Double,
                IsVisibleInLegend = true
            };

            foreach (RocPoint point in curve.Points)
            {
                rocSeries.Points.AddXY(point.Fpr, point.Tpr);
            }

            var rocPointsSeries = new Series
            {
                Name = "ROC Points",
                ChartType = SeriesChartType.Point,
                Color = ColorPalette[1],
                MarkerStyle = MarkerStyle.Circle,
                MarkerSize = 5,
                XValueType = ChartValueType.Double,
                YValueType = ChartValueType.Double,
                IsVisibleInLegend = true
            };

            // Keep the complete blue line but limit red markers so the
            // graph remains readable when many ROC points are returned
            int pointStep = Math.Max(1, curve.Points.Length / 25);

            for (int index = 0;
                 index < curve.Points.Length;
                 index += pointStep)
            {
                RocPoint point = curve.Points[index];

                rocPointsSeries.Points.AddXY(
                    point.Fpr,
                    point.Tpr);
            }

            var randomGuessSeries = new Series
            {
                Name = "Random Selection",
                ChartType = SeriesChartType.Line,
                Color = ColorPalette[2],
                BorderDashStyle = ChartDashStyle.Dash,
                BorderWidth = 2,
                XValueType = ChartValueType.Double,
                YValueType = ChartValueType.Double,
                IsVisibleInLegend = true
            };

            randomGuessSeries.Points.AddXY(0.0, 0.0);
            randomGuessSeries.Points.AddXY(1.0, 1.0);

            chart.Series.Add(rocSeries);
            chart.Series.Add(rocPointsSeries);
            chart.Series.Add(randomGuessSeries);

            chartArea.AxisX.Minimum = 0.0;
            chartArea.AxisX.Maximum = 1.0;
            chartArea.AxisY.Minimum = 0.0;
            chartArea.AxisY.Maximum = 1.0;

            chartArea.AxisX.Interval = 0.2;
            chartArea.AxisY.Interval = 0.2;

            chartArea.AxisX.LabelStyle.Format = "0.#";
            chartArea.AxisY.LabelStyle.Format = "0.#";

            chartArea.AxisX.LabelStyle.Font =
                new Font("Tahoma", 10);

            chartArea.AxisY.LabelStyle.Font =
                new Font("Tahoma", 10);

            chartArea.AxisX.MajorGrid.LineColor =
                Color.LightGray;

            chartArea.AxisY.MajorGrid.LineColor =
                Color.LightGray;

            chart.Legends.Add(new Legend
            {
                Docking = Docking.Bottom,
                Alignment = StringAlignment.Center,
                Font = new Font("Tahoma", 10),
                BackColor = Color.Transparent
            });

            _ = modelName;

            chart.Invalidate();

            // This is the AUC calculated by the NuGet package
            return curve.Score;
        }

        private static MetricsCurve<RocPoint> CalculateReceiverCurve(
            IEnumerable<float> scores,
            IEnumerable<bool> labels)
        {
            ArgumentNullException.ThrowIfNull(scores);
            ArgumentNullException.ThrowIfNull(labels);

            float[] scoreArray = scores.ToArray();
            bool[] labelArray = labels.ToArray();

            return CoreMetricsEvaluator.EvaluateRoc(
                scores: scoreArray,
                labels: labelArray,
                mode: RocMode.Exact,
                clampScoresToUnitInterval: false);
        }
    }
}