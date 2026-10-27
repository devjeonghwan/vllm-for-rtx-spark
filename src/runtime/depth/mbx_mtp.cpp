// Recovered from the pinned original AArch64 libmbx_mtp.so, not a new policy.
// Original SHA256 c0d164f84cf1161e430b53b161fe0b049316413263bf6fda250bd18792a95340.
// Address-to-statement evidence is recorded in recovery-map.json.
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
struct alignas(8) Configuration {
    std::int32_t maximum;
    std::int32_t minimum;
    std::int32_t window;
    std::int32_t padding;
    double promote_two;
    double promote_one;
    double demote_one;
    double demote_two;
};
static_assert(sizeof(Configuration) == 48);
static_assert(offsetof(Configuration, promote_two) == 16);
static_assert(sizeof(std::int32_t) == 4 && sizeof(std::int64_t) == 8);
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
Configuration configuration{5, 3, 48, 0, 60.0, 45.0, 25.0, 15.0};

std::int32_t signed32(std::uint32_t bits) noexcept {
    return std::bit_cast<std::int32_t>(bits);
}
std::int64_t signed64(std::uint64_t bits) noexcept {
    return std::bit_cast<std::int64_t>(bits);
}
std::int32_t clamp(std::int32_t value) noexcept {
    if (value < configuration.minimum) return configuration.minimum;
    if (value > configuration.maximum) return configuration.maximum;
    return value;
}
}  // namespace

// Native 0x650 leaves the configuration address in x0. Original Cython ignores
// the return value; uintptr_t additionally preserves that observed register.
extern "C" __declspec(dllexport) std::uintptr_t mbx_d_set(
    std::int32_t minimum, std::int32_t maximum, std::int32_t window,
    double promote_two, double promote_one, double demote_one,
    double demote_two) noexcept {
    configuration.maximum = maximum > 0 ? maximum : 1;
    configuration.minimum = minimum > 0
        ? (minimum < configuration.maximum ? minimum : configuration.maximum)
        : 1;
    configuration.window = window > 128 ? 128 : (window < 8 ? 8 : window);
    configuration.promote_two = promote_two;
    configuration.promote_one = promote_one;
    configuration.demote_one = demote_one;
    configuration.demote_two = demote_two;
    return reinterpret_cast<std::uintptr_t>(&configuration);
}

extern "C" __declspec(dllexport) std::int32_t mbx_d_start() noexcept {
    return configuration.minimum;  // Original 0x6a0.
}
extern "C" __declspec(dllexport) std::int32_t mbx_d_window() noexcept {
    return configuration.window;  // Original 0x6b0.
}
extern "C" __declspec(dllexport) std::int32_t mbx_d_ring() noexcept {
    return 128;  // Original 0x6c0.
}
extern "C" __declspec(dllexport) std::int32_t mbx_d_ready(
    std::int64_t position, std::int64_t last) noexcept {
    // Original 0x6d0: CMP signed X; SUB X wraps; CMP signed X against 23.
    if (position < static_cast<std::int64_t>(configuration.window)) return 0;
    const auto elapsed = signed64(static_cast<std::uint64_t>(position)
                                  - static_cast<std::uint64_t>(last));
    return elapsed > 23 ? 1 : 0;
}
extern "C" __declspec(dllexport) std::int32_t mbx_d_next(
    std::int32_t depth, double rate) noexcept {
    // Original 0x700: strict ordered comparisons; NaN takes no comparison arm.
    std::uint32_t candidate = static_cast<std::uint32_t>(depth);
    if (rate > configuration.promote_two) candidate += 2U;
    else if (rate > configuration.promote_one) candidate += 1U;
    else if (rate < configuration.demote_two) candidate -= 2U;
    else if (rate < configuration.demote_one) candidate -= 1U;
    return clamp(signed32(candidate));
}
extern "C" __declspec(dllexport) std::int32_t mbx_d_batch(
    std::int32_t count, const std::int32_t* depths,
    std::int32_t* outflag) noexcept {
    // Original 0x780: valid flag is required even when count<=0.
    *outflag = 0;
    if (count <= 0) return configuration.minimum;
    std::uint64_t sum_bits = 0;
    for (std::int32_t index = 0; index < count; ++index) {
        sum_bits += static_cast<std::uint64_t>(static_cast<std::int64_t>(depths[index]));
    }
    // 0x7b8 uses LSL W1 (32-bit), not an unbounded mathematical denominator.
    const auto denominator = static_cast<std::int64_t>(
        signed32(static_cast<std::uint32_t>(count) << 1));
    const auto numerator = signed64(static_cast<std::uint64_t>(count)
                                    + (sum_bits << 1));
    // For positive int32 count, denominator is nonzero and even; MIN/-1
    // cannot occur. C++ signed division therefore matches AArch64 SDIV.
    const auto quotient = numerator / denominator;
    // 0x7c8 compares W2: truncate the division result before signed clamp.
    return clamp(signed32(static_cast<std::uint32_t>(quotient)));
}
