using System.Runtime.InteropServices;

namespace ML.NET.Metrics.Evaluation.Public
{
    /// <summary>Represents one point on a receiver operating characteristic curve</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct RocPoint
    {
        /// <summary>False positive rate on the horizontal axis</summary>
        public double Fpr;

        /// <summary>True positive rate on the vertical axis</summary>
        public double Tpr;

        /// <summary>Score threshold represented by this point</summary>
        public float Threshold;
    }

    /// <summary>Selects the ROC calculation strategy</summary>
    public enum RocMode
    {
        /// <summary>Uses every unique score threshold and groups tied scores</summary>
        Exact,

        /// <summary>Uses fixed thresholds between one and zero</summary>
        Binned
    }

    /// <summary>Status values returned by the native coremetrics API</summary>
    public enum CoreMetricsStatus : int
    {
        /// <summary>The calculation completed successfully</summary>
        Ok = 0,

        /// <summary>A required native pointer was null</summary>
        ErrNullPtr = 1,

        /// <summary>The input length was zero</summary>
        ErrInvalidN = 2,

        /// <summary>The bucket count was invalid</summary>
        ErrInvalidBuckets = 3,

        /// <summary>The output buffer was too small</summary>
        ErrInvalidOutLen = 4,

        /// <summary>A label was not encoded as zero or one</summary>
        ErrInvalidLabel = 5,

        /// <summary>A score was NaN or infinity</summary>
        ErrNonfiniteScore = 6,

        /// <summary>The labels contained only one class</summary>
        ErrSingleClass = 7,

        /// <summary>A binned score was outside the unit interval</summary>
        ErrScoreOutOfRange = 8,

        /// <summary>Native memory allocation failed</summary>
        ErrAllocation = 9,

        /// <summary>An unexpected native failure occurred</summary>
        ErrInternal = 10,

        /// <summary>The requested native buffer size overflowed</summary>
        ErrSizeOverflow = 11
    }
}
