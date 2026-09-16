using System;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Windows.Forms.DataVisualization.Charting;

namespace ML.Graph
{
    /// <summary>
    /// renders previously calculated ROC points for binary classification
    /// </summary>
    public class BinaryChartDisplay : IChartDisplay
    {
        public void DisplayDataOnChart(Chart chart, DataTable dataTable, SeriesChartType chartType, string outcomeColumn)
        {
            if (chart == null)
                throw new ArgumentNullException(nameof(chart));
            if (dataTable == null)
                throw new ArgumentNullException(nameof(dataTable));
            if (string.IsNullOrWhiteSpace(outcomeColumn) ||
                !dataTable.Columns.Contains(outcomeColumn))
                throw new ArgumentException("Specify the true-positive-rate column.", nameof(outcomeColumn));
            if (!dataTable.Columns.Contains("FalsePositiveRate"))
                throw new ArgumentException("ROC data must contain a FalsePositiveRate column.", nameof(dataTable));
            if (dataTable.Columns[outcomeColumn] == dataTable.Columns["FalsePositiveRate"])
                throw new ArgumentException("False-positive and true-positive rates require separate columns.", nameof(outcomeColumn));
            if (dataTable.Rows.Count < 2)
                throw new ArgumentException("Provide a complete ROC curve with at least two points.", nameof(dataTable));

            var points = dataTable.AsEnumerable()
                .Select(row => new
                {
                    Fpr = ReadRate(row, "FalsePositiveRate"),
                    Tpr = ReadRate(row, outcomeColumn)
                })
                .OrderBy(point => point.Fpr)
                .ThenBy(point => point.Tpr)
                .ToArray();

            if (points[0].Fpr != 0 || points[0].Tpr != 0 ||
                points[points.Length - 1].Fpr != 1 || points[points.Length - 1].Tpr != 1)
                throw new ArgumentException("ROC points must include (0, 0) and (1, 1).", nameof(dataTable));

            for (int i = 1; i < points.Length; i++)
            {
                if (points[i].Tpr < points[i - 1].Tpr)
                    throw new ArgumentException("True-positive rates must not decrease as false-positive rates increase.", nameof(dataTable));
            }

            chart.Series.Clear();
            chart.ChartAreas.Clear();
            chart.Legends.Clear();

            var area = new ChartArea("ROC") { BackColor = Color.White };
            area.AxisX.Title = "False Positive Rate";
            area.AxisY.Title = "True Positive Rate";
            foreach (var axis in new[] { area.AxisX, area.AxisY })
            {
                axis.Minimum = 0;
                axis.Maximum = 1;
                axis.Interval = 0.2;
                axis.LabelStyle.Format = "0.##";
                axis.MajorGrid.LineColor = Color.LightGray;
                axis.TitleFont = new Font("Tahoma", 10);
            }
            chart.ChartAreas.Add(area);
            chart.Legends.Add(new Legend("ROC Legend")
            {
                Docking = Docking.Bottom,
                Alignment = StringAlignment.Center,
                BackColor = Color.Transparent,
                Font = new Font("Tahoma", 10)
            });

            var curve = new Series("ROC Curve")
            {
                ChartArea = area.Name,
                ChartType = SeriesChartType.Line,
                Color = Color.DodgerBlue,
                BorderWidth = 2,
                XValueType = ChartValueType.Double,
                YValueType = ChartValueType.Double
            };
            var markers = new Series("ROC Points")
            {
                ChartArea = area.Name,
                ChartType = SeriesChartType.Point,
                Color = Color.Tomato,
                MarkerStyle = MarkerStyle.Circle,
                MarkerSize = 5,
                XValueType = ChartValueType.Double,
                YValueType = ChartValueType.Double
            };
            var baseline = new Series("Random Selection")
            {
                ChartArea = area.Name,
                ChartType = SeriesChartType.Line,
                Color = Color.SpringGreen,
                BorderDashStyle = ChartDashStyle.Dash,
                BorderWidth = 2,
                XValueType = ChartValueType.Double,
                YValueType = ChartValueType.Double
            };
            foreach (var point in points)
            {
                curve.Points.AddXY(point.Fpr, point.Tpr);
                markers.Points.AddXY(point.Fpr, point.Tpr);
            }
            baseline.Points.AddXY(0.0, 0.0);
            baseline.Points.AddXY(1.0, 1.0);
            chart.Series.Add(curve);
            chart.Series.Add(markers);
            chart.Series.Add(baseline);
            chart.BackColor = Color.White;
            chart.Invalidate();
        }

        private static double ReadRate(DataRow row, string column)
        {
            object value = row[column];
            if (!(value is double || value is float || value is decimal ||
                  value is byte || value is sbyte || value is short ||
                  value is ushort || value is int || value is uint ||
                  value is long || value is ulong))
                throw new ArgumentException($"Column '{column}' must contain numeric ROC rates.");

            double rate = Convert.ToDouble(value, CultureInfo.InvariantCulture);
            if (double.IsNaN(rate) || double.IsInfinity(rate) || rate < 0 || rate > 1)
                throw new ArgumentException($"Column '{column}' must contain finite rates between 0 and 1.");
            return rate;
        }
    }
}
