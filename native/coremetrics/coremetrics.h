#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define CM_EXPORT __declspec(dllexport)
#define CM_CALL __cdecl
#else
#define CM_EXPORT __attribute__((visibility("default")))
#define CM_CALL
#endif

#if defined(__cplusplus)
#define CM_NOEXCEPT noexcept
extern "C" {
#else
#define CM_NOEXCEPT
#endif

// Stable status values used by the managed wrapper
enum CoreMetricsStatus : int32_t {
    CM_OK = 0,
    CM_ERR_NULLPTR = 1,
    CM_ERR_INVALID_N = 2,
    CM_ERR_INVALID_BUCKETS = 3,
    CM_ERR_INVALID_OUTLEN = 4,
    CM_ERR_INVALID_LABEL = 5,
    CM_ERR_NONFINITE_SCORE = 6,
    CM_ERR_SINGLE_CLASS = 7,
    CM_ERR_SCORE_OUT_OF_RANGE = 8,
    CM_ERR_ALLOCATION = 9,
    CM_ERR_INTERNAL = 10,
    CM_ERR_SIZE_OVERFLOW = 11
};

struct RocPoint {
    double Fpr;
    double Tpr;
    float Threshold;
};

// Fixed threshold approximation for probability like scores. Thresholds are
// swept from 1.0 down to 0.0 and buckets + 1 points are written
//
// If clamp_scores_to_unit_interval is zero, every score must already be in
// [0, 1]. If it is one, finite scores are clamped before sorting/bucketing
CM_EXPORT int32_t CM_CALL ComputeRocAuc_Binned(
    const float* scores,
    const uint8_t* labels,
    size_t n,
    int32_t buckets,
    RocPoint* out_roc,
    size_t out_roc_len,
    double* out_auc,
    uint8_t clamp_scores_to_unit_interval
) CM_NOEXCEPT;

// Exact ROC: origin, one point per unique score (ties are handled as one
// group), and an explicit final endpoint. On CM_ERR_INVALID_OUTLEN,
// out_points_written receives the required capacity
CM_EXPORT int32_t CM_CALL ComputeRocAuc_Exact(
    const float* scores,
    const uint8_t* labels,
    size_t n,
    RocPoint* out_roc,
    size_t out_roc_len,
    size_t* out_points_written,
    double* out_auc,
    uint8_t clamp_scores_to_unit_interval
) CM_NOEXCEPT;

// Exact AUC without allocating or returning the ROC point buffer
CM_EXPORT int32_t CM_CALL ComputeAuc_Exact(
    const float* scores,
    const uint8_t* labels,
    size_t n,
    double* out_auc,
    uint8_t clamp_scores_to_unit_interval
) CM_NOEXCEPT;

#if defined(__cplusplus)
} // extern "C"
#endif
