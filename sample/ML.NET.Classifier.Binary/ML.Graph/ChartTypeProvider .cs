using System.Windows.Forms.DataVisualization.Charting;

namespace ML.Graph
{
    /// <summary>
    /// provides the ROC chart option for binary classification
    /// </summary>
    public class ChartTypeProvider : IChartTypeProvider
    {
        // parameters are retained for existing form calls
        public SeriesChartType GetDefaultChartType(string classificationResult)
        {
            return SeriesChartType.Line;
        }

        public string[] GetAvailableChartTypes(string classificationResult)
        {
            return new[] { "Receiver operating characteristic curve" };
        }

        public SeriesChartType GetChartTypeFromName(string chartName)
        {
            return SeriesChartType.Line;
        }
    }
}
