using ML.NET.Metrics.Evaluation.Internal;

namespace ML.NET.Metrics.Evaluation.Public
{
    /// <summary>Computes exact or fixed threshold ROC curves and AUC values</summary>
    public static class CoreMetricsEvaluator
    {
        private const int MaxBinnedBuckets = 1_000_000;

        /// <summary>
        /// Gets whether the current process can load the packaged native runtime
        /// Version 0.5.0 supports Windows x64 processes
        /// </summary>
        public static bool IsNativeRuntimeSupported => NativeResolver.IsSupportedPlatform;

        /// <summary>
        /// Computes a ROC curve and its matching AUC from aligned ranking scores and labels
        /// Exact mode accepts any finite score range. Binned mode requires scores in [0,1]
        /// unless clamping is explicitly enabled
        /// </summary>
        public static MetricsCurve<RocPoint> EvaluateRoc(
            float[] scores,
            bool[] labels,
            RocMode mode = RocMode.Exact,
            int buckets = 100,
            bool clampScoresToUnitInterval = false)
        {
            NativeResolver.EnsureRegistered();

            if (!Enum.IsDefined(typeof(RocMode), mode))
                throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unsupported ROC mode.");

            byte[] labels8 = ValidateAndConvertInputs(scores, labels);
            byte clamp01 = clampScoresToUnitInterval ? (byte)1 : (byte)0;

            if (mode == RocMode.Binned)
            {
                ValidateBinnedOptions(scores, buckets, clampScoresToUnitInterval);
                int bins = checked(buckets + 1);
                var nativeRoc = new RocPoint[bins];

                int status = CoreMetricsMethods.ComputeRocAuc_Binned(
                    scores,
                    labels8,
                    (nuint)scores.Length,
                    buckets,
                    nativeRoc,
                    (nuint)nativeRoc.Length,
                    out double auc,
                    clamp01);

                EnsureSuccess(status, "binned ROC/AUC");

                // Native binned output contains thresholds from 1 to 0. Prepend
                // the conceptual threshold above all scores so the managed curve
                // contains the same origin that participates in native AUC
                var roc = new RocPoint[checked(nativeRoc.Length + 1)];
                roc[0] = new RocPoint
                {
                    Fpr = 0.0,
                    Tpr = 0.0,
                    Threshold = float.PositiveInfinity
                };
                Array.Copy(nativeRoc, 0, roc, 1, nativeRoc.Length);

                return new MetricsCurve<RocPoint>(roc, auc);
            }

            int maximumPointCount = checked(scores.Length + 2);
            var rocBuffer = new RocPoint[maximumPointCount];
            int exactStatus = CoreMetricsMethods.ComputeRocAuc_Exact(
                scores,
                labels8,
                (nuint)scores.Length,
                rocBuffer,
                (nuint)rocBuffer.Length,
                out nuint written,
                out double exactAuc,
                clamp01);

            EnsureSuccess(exactStatus, "exact ROC/AUC");
            int writtenCount = checked((int)written);
            if (writtenCount < 3 || writtenCount > rocBuffer.Length)
                throw new InvalidOperationException("The native library returned an invalid ROC point count.");

            var exactRoc = new RocPoint[writtenCount];
            Array.Copy(rocBuffer, exactRoc, writtenCount);
            return new MetricsCurve<RocPoint>(exactRoc, exactAuc);
        }

        /// <summary>
        /// Computes exact AUC without allocating a managed ROC point array. Scores may be
        /// probabilities or unrestricted finite decision scores
        /// </summary>
        public static double EvaluateAuc(
            float[] scores,
            bool[] labels,
            bool clampScoresToUnitInterval = false)
        {
            NativeResolver.EnsureRegistered();

            byte[] labels8 = ValidateAndConvertInputs(scores, labels);
            int status = CoreMetricsMethods.ComputeAuc_Exact(
                scores,
                labels8,
                (nuint)scores.Length,
                out double auc,
                clampScoresToUnitInterval ? (byte)1 : (byte)0);

            EnsureSuccess(status, "exact AUC");
            return auc;
        }

        private static byte[] ValidateAndConvertInputs(float[] scores, bool[] labels)
        {
            ArgumentNullException.ThrowIfNull(scores);
            ArgumentNullException.ThrowIfNull(labels);
            if (scores.Length != labels.Length)
                throw new ArgumentException("Scores and labels must have the same length.");
            if (scores.Length == 0)
                throw new ArgumentException("Scores and labels cannot be empty.");

            bool hasPositive = false;
            bool hasNegative = false;
            var labels8 = new byte[labels.Length];
            for (int i = 0; i < scores.Length; ++i)
            {
                if (!float.IsFinite(scores[i]))
                    throw new ArgumentException($"Score at index {i} is not finite.", nameof(scores));

                labels8[i] = labels[i] ? (byte)1 : (byte)0;
                hasPositive |= labels[i];
                hasNegative |= !labels[i];
            }

            if (!hasPositive || !hasNegative)
                throw new ArgumentException("ROC AUC is undefined when labels contain only one class.", nameof(labels));

            return labels8;
        }

        private static void ValidateBinnedOptions(
            float[] scores,
            int buckets,
            bool clampScoresToUnitInterval)
        {
            if (buckets <= 0 || buckets > MaxBinnedBuckets)
                throw new ArgumentOutOfRangeException(
                    nameof(buckets), buckets, $"Buckets must be between 1 and {MaxBinnedBuckets:N0}.");

            if (clampScoresToUnitInterval) return;
            for (int i = 0; i < scores.Length; ++i)
            {
                if (scores[i] < 0.0f || scores[i] > 1.0f)
                    throw new ArgumentOutOfRangeException(
                        nameof(scores),
                        $"Binned mode requires scores in [0,1]; index {i} contains {scores[i]}. " +
                        "Use Exact mode for raw decision scores, or enable clamping deliberately.");
            }
        }

        private static void EnsureSuccess(int status, string operation)
        {
            var code = (CoreMetricsStatus)status;
            switch (code)
            {
                case CoreMetricsStatus.Ok:
                    return;
                case CoreMetricsStatus.ErrNullPtr:
                    throw new InvalidOperationException($"The native {operation} call received a null pointer.");
                case CoreMetricsStatus.ErrInvalidN:
                    throw new ArgumentException("Scores and labels cannot be empty.", "scores");
                case CoreMetricsStatus.ErrInvalidBuckets:
                    throw new ArgumentOutOfRangeException("buckets", "Buckets must be greater than zero.");
                case CoreMetricsStatus.ErrInvalidOutLen:
                    throw new InvalidOperationException($"The native {operation} output buffer was too small.");
                case CoreMetricsStatus.ErrInvalidLabel:
                    throw new ArgumentException("Labels must be encoded as zero or one.", "labels");
                case CoreMetricsStatus.ErrNonfiniteScore:
                    throw new ArgumentException("All scores must be finite.", "scores");
                case CoreMetricsStatus.ErrSingleClass:
                    throw new ArgumentException("ROC AUC requires both positive and negative labels.", "labels");
                case CoreMetricsStatus.ErrScoreOutOfRange:
                    throw new ArgumentOutOfRangeException(
                        "scores", "Binned ROC requires scores in [0,1] unless clamping is enabled.");
                case CoreMetricsStatus.ErrAllocation:
                    throw new OutOfMemoryException($"The native {operation} calculation could not allocate memory.");
                case CoreMetricsStatus.ErrSizeOverflow:
                    throw new OverflowException($"The native {operation} input size is unsupported.");
                case CoreMetricsStatus.ErrInternal:
                    throw new InvalidOperationException($"The native {operation} calculation failed internally.");
                default:
                    throw new InvalidOperationException($"Native {operation} failed with status {code} ({status}).");
            }
        }
    }
}
