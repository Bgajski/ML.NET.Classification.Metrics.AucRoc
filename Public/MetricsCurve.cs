namespace ML.NET.Metrics.Evaluation.Public
{
    /// <summary>Contains metric curve points and the scalar score calculated from them</summary>
    /// <typeparam name="TPoint">Type of point contained in the curve</typeparam>
    public sealed class MetricsCurve<TPoint>
    {
        /// <summary>Creates a curve result</summary>
        /// <param name="points">Ordered points forming the curve</param>
        /// <param name="score">Scalar metric associated with the curve</param>
        public MetricsCurve(TPoint[] points, double score)
        {
            ArgumentNullException.ThrowIfNull(points);
            Points = points;
            Score = score;
        }

        /// <summary>Gets the ordered curve points</summary>
        public TPoint[] Points { get; }

        /// <summary>Gets the scalar metric, for a ROC curve this is AUC</summary>
        public double Score { get; }
    }
}
