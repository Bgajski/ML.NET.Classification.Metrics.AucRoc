#include "coremetrics.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

    // Internal representation of one observation
    //
    // OriginalIndex is retained so samples with identical scores have a stable,
    // deterministic order after sorting
    struct RankedSample {
        float Score;
        uint8_t Label;
        size_t OriginalIndex;
    };

    // Restricts a score to the inclusive [0, 1] interval
    //
    // This is used only when the caller enables score clamping
    float clamp01(float value) noexcept {
        if (value < 0.0f) return 0.0f;
        if (value > 1.0f) return 1.0f;
        return value;
    }

    // Adds the area between two neighboring ROC points using the trapezoidal rule
    //
    // ROC points are ordered by increasing false positive rate. Vertical movements
    // have delta_fpr == 0 and therefore contribute no area
    void add_trapezoid(
        double& auc,
        const RocPoint& previous,
        const RocPoint& current
    ) noexcept {
        const double delta_fpr = current.Fpr - previous.Fpr;

        if (delta_fpr > 0.0) {
            auc += delta_fpr * (current.Tpr + previous.Tpr) * 0.5;
        }
    }

    // Validates the input arrays, optionally normalizes scores, counts both
    // classes, and sorts samples from the highest score to the lowest
    //
    // Parameters:
    //   clamp_scores
    //       If true, scores outside [0, 1] are clamped before sorting
    //
    //   require_unit_interval
    //       If true and clamping is disabled, every score must already be within
    //       [0, 1]. This is required by the binned ROC implementation because its
    //       thresholds are fixed to that interval
    //
    // Exact ROC/AUC calculation does not require scores to represent probabilities,
    // any finite numeric ranking score is valid
    int32_t build_ranked_samples(
        const float* scores,
        const uint8_t* labels,
        size_t n,
        bool clamp_scores,
        bool require_unit_interval,
        std::vector<RankedSample>& samples,
        uint64_t& total_positive,
        uint64_t& total_negative
    ) {
        // Reserve all required storage up front to avoid repeated reallocations
        samples.reserve(n);

        total_positive = 0;
        total_negative = 0;

        for (size_t i = 0; i < n; ++i) {
            // Binary classification labels must be exactly 0 or 1
            if (labels[i] > 1u) {
                return CM_ERR_INVALID_LABEL;
            }

            const float original_score = scores[i];

            // NaN and positive/negative infinity cannot be reliably ranked
            if (!std::isfinite(original_score)) {
                return CM_ERR_NONFINITE_SCORE;
            }

            // Binned ROC thresholds cover only [0, 1]. If clamping is disabled,
            // reject values that lie outside that interval
            if (require_unit_interval &&
                !clamp_scores &&
                (original_score < 0.0f || original_score > 1.0f)) {
                return CM_ERR_SCORE_OUT_OF_RANGE;
            }

            const float score =
                clamp_scores ? clamp01(original_score) : original_score;

            samples.push_back(RankedSample{
                score,
                labels[i],
                i
                });

            if (labels[i] == 1u) {
                ++total_positive;
            }
            else {
                ++total_negative;
            }
        }

        // ROC/AUC is undefined when the dataset contains only one class
        if (total_positive == 0 || total_negative == 0) {
            return CM_ERR_SINGLE_CLASS;
        }

        // ROC construction processes observations from the highest score to the
        // lowest score. Equal score observations are ordered by their original
        // position to make sorting deterministic
        //
        // The exact calculation still processes an entire equal score group at
        // once, so the order inside a tie does not affect the resulting AUC
        std::sort(
            samples.begin(),
            samples.end(),
            [](const RankedSample& left, const RankedSample& right) {
                if (left.Score == right.Score) {
                    return left.OriginalIndex < right.OriginalIndex;
                }

                return left.Score > right.Score;
            });

        return CM_OK;
    }

    // Calculates exact AUC from samples already sorted by descending score
    //
    // Every group of equal scores is processed together. This is important because
    // choosing an arbitrary order inside a tie would incorrectly give one tied
    // sample precedence over another and could change the AUC
    double exact_auc_from_sorted(
        const std::vector<RankedSample>& samples,
        uint64_t total_positive,
        uint64_t total_negative
    ) noexcept {
        uint64_t true_positive = 0;
        uint64_t false_positive = 0;
        size_t cursor = 0;
        double auc = 0.0;

        // A threshold above every finite score predicts all samples as negative,
        // producing the initial ROC point (FPR=0, TPR=0)
        RocPoint previous{
            0.0,
            0.0,
            std::numeric_limits<float>::infinity()
        };

        while (cursor < samples.size()) {
            const float threshold = samples[cursor].Score;
            size_t next = cursor;

            // Include all observations tied at the current threshold before
            // producing the next ROC point
            while (next < samples.size() &&
                samples[next].Score == threshold) {
                if (samples[next].Label == 1u) {
                    ++true_positive;
                }
                else {
                    ++false_positive;
                }

                ++next;
            }

            const RocPoint current{
                static_cast<double>(false_positive) /
                    static_cast<double>(total_negative),

                static_cast<double>(true_positive) /
                    static_cast<double>(total_positive),

                threshold
            };

            add_trapezoid(auc, previous, current);

            previous = current;
            cursor = next;
        }

        // After processing the lowest unique score, all observations have been
        // classified as positive, so the curve has already reached (1, 1)
        //
        // The explicit infinity threshold preserves the public ROC curve format
        // Since it duplicates the final coordinates, it adds no extra area
        const RocPoint endpoint{
            1.0,
            1.0,
            -std::numeric_limits<float>::infinity()
        };

        add_trapezoid(auc, previous, endpoint);

        return auc;
    }

    // Counts the number of distinct score values in an already sorted sample list
    //
    // The exact ROC curve contains:
    //   1 initial + infinity point,
    //   1 point for every unique score,
    //   1 final - infinity point
    size_t unique_score_count(
        const std::vector<RankedSample>& samples
    ) noexcept {
        size_t count = 0;
        size_t cursor = 0;

        while (cursor < samples.size()) {
            ++count;

            const float score = samples[cursor].Score;

            // Skip the entire group of samples sharing this score
            do {
                ++cursor;
            } while (
                cursor < samples.size() &&
                samples[cursor].Score == score
                );
        }

        return count;
    }

    // Prevents C++ exceptions from crossing the native C ABI boundary
    //
    // Every exported function is declared noexcept and returns an explicit
    // CoreMetrics error code instead of propagating an exception to the caller
    template <typename Function>
    int32_t protect_native_boundary(Function&& function) noexcept {
        try {
            return function();
        }
        catch (const std::bad_alloc&) {
            return CM_ERR_ALLOCATION;
        }
        catch (const std::length_error&) {
            return CM_ERR_SIZE_OVERFLOW;
        }
        catch (...) {
            return CM_ERR_INTERNAL;
        }
    }

}

// Builds an approximate ROC curve using evenly spaced thresholds in [0, 1]
// and calculates its trapezoidal AUC
//
// With B buckets, the function writes B + 1 points using thresholds:
//
//   1.0, 1.0 - 1/B, ..., 1/B, 0.0
//
// Because the thresholds are restricted to [0, 1], input scores must either
// already lie inside that interval or be clamped by the function
extern "C" int32_t CM_CALL ComputeRocAuc_Binned(
    const float* scores,
    const uint8_t* labels,
    size_t n,
    int32_t buckets,
    RocPoint* out_roc,
    size_t out_roc_len,
    double* out_auc,
    uint8_t clamp_scores_to_unit_interval
) noexcept {
    // Initialize output early so callers never receive an uninitialized AUC
    // when validation or computation fails
    if (out_auc) {
        *out_auc = 0.0;
    }

    if (!scores || !labels || !out_roc || !out_auc) {
        return CM_ERR_NULLPTR;
    }

    if (n == 0) {
        return CM_ERR_INVALID_N;
    }

    if (buckets <= 0) {
        return CM_ERR_INVALID_BUCKETS;
    }

    // Both ends of the interval are included, hence buckets + 1 points
    const size_t point_count =
        static_cast<size_t>(buckets) + 1u;

    if (out_roc_len < point_count) {
        return CM_ERR_INVALID_OUTLEN;
    }

    return protect_native_boundary([&]() -> int32_t {
        std::vector<RankedSample> samples;
        uint64_t total_positive = 0;
        uint64_t total_negative = 0;

        const int32_t validation = build_ranked_samples(
            scores,
            labels,
            n,
            clamp_scores_to_unit_interval != 0,
            true, // Binned calculation requires scores in [0, 1]
            samples,
            total_positive,
            total_negative
        );

        if (validation != CM_OK) {
            return validation;
        }

        uint64_t true_positive = 0;
        uint64_t false_positive = 0;
        size_t cursor = 0;
        double auc = 0.0;

        // Conceptual starting point: the threshold is above every score, so
        // no observation is predicted as positive
        RocPoint previous{
            0.0,
            0.0,
            std::numeric_limits<float>::infinity()
        };

        for (size_t bucket = 0; bucket < point_count; ++bucket) {
            // Generate a threshold sequence from 1.0 down to 0.0
            const float threshold = static_cast<float>(
                1.0 -
                static_cast<double>(bucket) /
                static_cast<double>(buckets)
                );

            // Samples are sorted in descending order. Advance the cursor only
            // for samples newly included at this threshold
            while (
                cursor < samples.size() &&
                samples[cursor].Score >= threshold
                ) {
                if (samples[cursor].Label == 1u) {
                    ++true_positive;
                }
                else {
                    ++false_positive;
                }

                ++cursor;
            }

            const RocPoint current{
                static_cast<double>(false_positive) /
                    static_cast<double>(total_negative),

                static_cast<double>(true_positive) /
                    static_cast<double>(total_positive),

                threshold
            };

            out_roc[bucket] = current;

            // Integrate the segment connecting the previous and current
            // binned ROC points
            add_trapezoid(auc, previous, current);

            previous = current;
        }

        *out_auc = auc;
        return CM_OK;
        });
}

// Builds the exact empirical ROC curve and calculates its AUC
//
// One ROC point is generated after every distinct score threshold. Equal score
// samples are processed as a group
//
// On CM_ERR_INVALID_OUTLEN, out_points_written contains the required output
// capacity so the caller can allocate a sufficiently large buffer and retry
extern "C" int32_t CM_CALL ComputeRocAuc_Exact(
    const float* scores,
    const uint8_t* labels,
    size_t n,
    RocPoint* out_roc,
    size_t out_roc_len,
    size_t* out_points_written,
    double* out_auc,
    uint8_t clamp_scores_to_unit_interval
) noexcept {
    // Initialize output parameters before validating the remaining inputs
    if (out_points_written) {
        *out_points_written = 0;
    }

    if (out_auc) {
        *out_auc = 0.0;
    }

    if (!scores ||
        !labels ||
        !out_roc ||
        !out_points_written ||
        !out_auc) {
        return CM_ERR_NULLPTR;
    }

    if (n == 0) {
        return CM_ERR_INVALID_N;
    }

    // The exact curve may require up to n + 2 points:
    // one per unique score plus the two endpoint thresholds
    if (n > std::numeric_limits<size_t>::max() - 2u) {
        return CM_ERR_SIZE_OVERFLOW;
    }

    return protect_native_boundary([&]() -> int32_t {
        std::vector<RankedSample> samples;
        uint64_t total_positive = 0;
        uint64_t total_negative = 0;

        const int32_t validation = build_ranked_samples(
            scores,
            labels,
            n,
            clamp_scores_to_unit_interval != 0,
            false, // Exact ROC accepts any finite score range
            samples,
            total_positive,
            total_negative
        );

        if (validation != CM_OK) {
            return validation;
        }

        // Exact output contains one point per unique score and two explicit
        // endpoint points at + infinity and - infinity
        const size_t required =
            unique_score_count(samples) + 2u;

        // Report required capacity even if the supplied buffer is too small
        *out_points_written = required;

        if (out_roc_len < required) {
            return CM_ERR_INVALID_OUTLEN;
        }

        size_t written = 0;
        
        // A threshold of + infinity predicts no finite scored sample as
        // positive, giving the initial ROC point (0, 0)
        out_roc[written++] = RocPoint{
            0.0,
            0.0,
            std::numeric_limits<float>::infinity()
        };

        uint64_t true_positive = 0;
        uint64_t false_positive = 0;
        size_t cursor = 0;
        double auc = 0.0;

        RocPoint previous = out_roc[0];

        while (cursor < samples.size()) {
            const float threshold = samples[cursor].Score;
            size_t next = cursor;

            // Move every observation tied at this threshold into the
            // predicted positive group at the same time
            while (
                next < samples.size() &&
                samples[next].Score == threshold
                ) {
                if (samples[next].Label == 1u) {
                    ++true_positive;
                }
                else {
                    ++false_positive;
                }

                ++next;
            }

            const RocPoint current{
                static_cast<double>(false_positive) /
                    static_cast<double>(total_negative),

                static_cast<double>(true_positive) /
                    static_cast<double>(total_positive),

                threshold
            };

            out_roc[written++] = current;

            // Add the area of the newly created ROC segment
            add_trapezoid(auc, previous, current);

            previous = current;
            cursor = next;
        }

        // A threshold of - infinity predicts every sample as positive
        const RocPoint endpoint{
            1.0,
            1.0,
            -std::numeric_limits<float>::infinity()
        };

        out_roc[written++] = endpoint;
        add_trapezoid(auc, previous, endpoint);

        *out_points_written = written;
        *out_auc = auc;

        return CM_OK;
        });
}

// Calculates exact AUC without returning the individual ROC points
//
// This is more convenient when the caller needs only the final AUC value,
// although samples must still be sorted to obtain the exact result
extern "C" int32_t CM_CALL ComputeAuc_Exact(
    const float* scores,
    const uint8_t* labels,
    size_t n,
    double* out_auc,
    uint8_t clamp_scores_to_unit_interval
) noexcept {
    // Keep the output deterministic when validation or allocation fails
    if (out_auc) {
        *out_auc = 0.0;
    }

    if (!scores || !labels || !out_auc) {
        return CM_ERR_NULLPTR;
    }

    if (n == 0) {
        return CM_ERR_INVALID_N;
    }

    return protect_native_boundary([&]() -> int32_t {
        std::vector<RankedSample> samples;
        uint64_t total_positive = 0;
        uint64_t total_negative = 0;

        const int32_t validation = build_ranked_samples(
            scores,
            labels,
            n,
            clamp_scores_to_unit_interval != 0,
            false, // Exact AUC accepts any finite score range
            samples,
            total_positive,
            total_negative
        );

        if (validation != CM_OK) {
            return validation;
        }

        *out_auc = exact_auc_from_sorted(
            samples,
            total_positive,
            total_negative
        );

        return CM_OK;
        });
}
