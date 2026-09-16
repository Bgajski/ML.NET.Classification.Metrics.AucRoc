using ML.NET.Metrics.Evaluation.Public;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ML.NET.Metrics.Evaluation.Internal
{
    internal static partial class CoreMetricsMethods
    {
        private const string Dll = "coremetrics";

        [LibraryImport(Dll, EntryPoint = "ComputeRocAuc_Binned")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial int ComputeRocAuc_Binned(
            float[] scores,
            byte[] labels,
            nuint n,
            int buckets,
            [Out] RocPoint[] rocOut,
            nuint rocOutLen,
            out double auc,
            byte clamp01);

        [LibraryImport(Dll, EntryPoint = "ComputeRocAuc_Exact")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial int ComputeRocAuc_Exact(
            float[] scores,
            byte[] labels,
            nuint n,
            [Out] RocPoint[] rocOut,
            nuint rocOutLen,
            out nuint pointsWritten,
            out double auc,
            byte clamp01);

        [LibraryImport(Dll, EntryPoint = "ComputeAuc_Exact")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        internal static partial int ComputeAuc_Exact(
            float[] scores,
            byte[] labels,
            nuint n,
            out double auc,
            byte clamp01);
    }
}
