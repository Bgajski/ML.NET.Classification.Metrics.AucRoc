using System.Runtime.InteropServices;

namespace ML.NET.Metrics.Evaluation.Internal
{
    /// <summary>Resolves the platform specific coremetrics native library</summary>
    internal static class NativeResolver
    {
        private static readonly object InitializationLock = new();
        private static bool _initialized;

        internal static bool IsSupportedPlatform =>
            OperatingSystem.IsWindows() &&
            RuntimeInformation.ProcessArchitecture == Architecture.X64;

        internal static void EnsureRegistered()
        {
            if (!IsSupportedPlatform)
            {
                throw new PlatformNotSupportedException(
                    "ML.NET.Classification.Metrics.AucRoc 0.5.0 supports Windows x64 processes only.");
            }

            lock (InitializationLock)
            {
                if (_initialized) return;
                NativeLibrary.SetDllImportResolver(typeof(NativeResolver).Assembly, Resolve);
                _initialized = true;
            }
        }

        private static IntPtr Resolve(
            string libraryName,
            System.Reflection.Assembly assembly,
            DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, "coremetrics", StringComparison.OrdinalIgnoreCase))
                return IntPtr.Zero;

            const string fileName = "coremetrics.dll";

            if (NativeLibrary.TryLoad(fileName, assembly, searchPath, out IntPtr handle))
                return handle;

            string candidate = Path.Combine(AppContext.BaseDirectory, fileName);
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out handle))
                return handle;

            return IntPtr.Zero;
        }
    }
}
