// Deterministic CUDA/CPU reference oracle for directional visible reflectance
// from a homogeneous, plane-parallel cloud slab over a black lower boundary.
//
// This is deliberately standalone. It is a scientific calibration experiment,
// not a runtime dependency of the production renderer.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define HD __host__ __device__

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kThreads = 256;
constexpr int kStage2DepthBins = 32;
constexpr double kInvU32 = 2.3283064365386962890625e-10;  // 2^-32

struct Vec3 {
    double x;
    double y;
    double z;
};

HD Vec3 make_vec(double x, double y, double z) { return {x, y, z}; }
HD Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
HD Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
HD Vec3 operator*(Vec3 a, double b) { return {a.x * b, a.y * b, a.z * b}; }
HD double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
HD Vec3 normalize(Vec3 a) {
    const double inv = 1.0 / sqrt(dot(a, a));
    return a * inv;
}

struct U32x4 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

HD uint32_t mul_hi_u32(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) >> 32);
}

HD U32x4 philox_round(U32x4 c, uint32_t k0, uint32_t k1) {
    constexpr uint32_t m0 = 0xD2511F53U;
    constexpr uint32_t m1 = 0xCD9E8D57U;
    const uint32_t lo0 = m0 * c.x;
    const uint32_t hi0 = mul_hi_u32(m0, c.x);
    const uint32_t lo1 = m1 * c.z;
    const uint32_t hi1 = mul_hi_u32(m1, c.z);
    return {hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
}

HD U32x4 philox4x32_10(U32x4 counter, uint32_t key0, uint32_t key1) {
    constexpr uint32_t w0 = 0x9E3779B9U;
    constexpr uint32_t w1 = 0xBB67AE85U;
    for (int round = 0; round < 10; ++round) {
        counter = philox_round(counter, key0, key1);
        if (round != 9) {
            key0 += w0;
            key1 += w1;
        }
    }
    return counter;
}

struct CounterRng {
    uint64_t path;
    uint64_t seed;
    uint64_t block = 0;
    int lane = 4;
    U32x4 words{};

    HD CounterRng(uint64_t path_index, uint64_t seed_value)
        : path(path_index), seed(seed_value) {}

    HD void refill() {
        const U32x4 counter{static_cast<uint32_t>(path), static_cast<uint32_t>(path >> 32),
                            static_cast<uint32_t>(block), static_cast<uint32_t>(block >> 32)};
        words = philox4x32_10(counter, static_cast<uint32_t>(seed),
                              static_cast<uint32_t>(seed >> 32));
        ++block;
        lane = 0;
    }

    HD uint32_t next_u32() {
        if (lane == 4) refill();
        const uint32_t values[4] = {words.x, words.y, words.z, words.w};
        return values[lane++];
    }

    HD double uniform_open() {
        return (static_cast<double>(next_u32()) + 0.5) * kInvU32;
    }
};

struct Parameters {
    std::string case_name = "single";
    double tau = 2.0;
    double ssa = 0.999;
    double g = 0.85;
    double sun_zenith_deg = 30.0;
    double view_zenith_deg = 0.0;
    double relative_azimuth_deg = 0.0;
    uint64_t samples = 262144;
    uint64_t seed = 0x53415453494D0015ULL;
    int max_scatters = 128;
    uint64_t batch_samples = 65536;
    std::string phase_model = "single-hg";
    double phase_lobe2_g = 0.0;
    double phase_lobe1_weight = 1.0;
    double expected_phase_first_moment = std::numeric_limits<double>::quiet_NaN();
    double surface_albedo = 0.0;
    std::string lower_boundary = "lambertian";
    int depth_bins = 0;
    bool report_forward_flux = false;
    bool stage2_requested = false;
};

struct Model {
    double tau;
    double ssa;
    double g;
    double mu0;
    double muv;
    Vec3 sun_direction;
    Vec3 initial_backward;
    int max_scatters;
};

struct Stage2Model {
    double tau;
    double ssa;
    double phase_lobe1_g;
    double phase_lobe2_g;
    double phase_lobe1_weight;
    double surface_albedo;
    double mu0;
    double muv;
    Vec3 sun_direction;
    Vec3 initial_backward;
    int max_scatters;
};

Model make_model(const Parameters& p) {
    const double sun = p.sun_zenith_deg * kPi / 180.0;
    const double view = p.view_zenith_deg * kPi / 180.0;
    const double az = p.relative_azimuth_deg * kPi / 180.0;
    const double mu0 = std::cos(sun);
    const double muv = std::cos(view);

    // Relative azimuth is defined here between the incoming photon's horizontal
    // propagation direction and the outgoing view direction.
    const Vec3 sun_direction{std::sin(sun), 0.0, -mu0};
    const Vec3 view_out{std::sin(view) * std::cos(az), std::sin(view) * std::sin(az), muv};
    return {p.tau, p.ssa, p.g, mu0, muv, sun_direction, -view_out, p.max_scatters};
}

Stage2Model make_stage2_model(const Parameters& p) {
    const Model legacy = make_model(p);
    return {legacy.tau,
            legacy.ssa,
            p.g,
            p.phase_lobe2_g,
            p.phase_lobe1_weight,
            p.surface_albedo,
            legacy.mu0,
            legacy.muv,
            legacy.sun_direction,
            legacy.initial_backward,
            legacy.max_scatters};
}

HD double clamp_unit(double x) { return x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x); }

HD double hg_phase(double cosine, double g) {
    cosine = clamp_unit(cosine);
    const double gg = g * g;
    const double denominator = 1.0 + gg - 2.0 * g * cosine;
    return (1.0 - gg) / (4.0 * kPi * denominator * sqrt(denominator));
}

HD double mixture_hg_phase(double cosine, const Stage2Model& model) {
    if (model.phase_lobe1_weight >= 1.0) {
        return hg_phase(cosine, model.phase_lobe1_g);
    }
    if (model.phase_lobe1_weight <= 0.0) {
        return hg_phase(cosine, model.phase_lobe2_g);
    }
    return model.phase_lobe1_weight * hg_phase(cosine, model.phase_lobe1_g) +
           (1.0 - model.phase_lobe1_weight) * hg_phase(cosine, model.phase_lobe2_g);
}

HD Vec3 sample_hg(Vec3 axis, double g, CounterRng& rng) {
    const double u = rng.uniform_open();
    double cosine;
    if (fabs(g) < 1.0e-8) {
        cosine = 2.0 * u - 1.0;
    } else {
        const double ratio = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
        cosine = (1.0 + g * g - ratio * ratio) / (2.0 * g);
        cosine = clamp_unit(cosine);
    }
    const double sine = sqrt(fmax(0.0, 1.0 - cosine * cosine));
    const double phi = 2.0 * kPi * rng.uniform_open();

    Vec3 tangent;
    if (fabs(axis.z) < 0.999999) {
        tangent = normalize(cross(make_vec(0.0, 0.0, 1.0), axis));
    } else {
        tangent = normalize(cross(make_vec(0.0, 1.0, 0.0), axis));
    }
    const Vec3 bitangent = cross(axis, tangent);
    return normalize(axis * cosine + tangent * (sine * cos(phi)) + bitangent * (sine * sin(phi)));
}

HD Vec3 sample_mixture_hg(Vec3 axis, const Stage2Model& model, CounterRng& rng) {
    // The identity branches deliberately consume exactly the legacy two HG
    // draws.  That keeps black-surface/single-HG Stage-1 paths bit-identical.
    if (model.phase_lobe1_weight >= 1.0) {
        return sample_hg(axis, model.phase_lobe1_g, rng);
    }
    if (model.phase_lobe1_weight <= 0.0) {
        return sample_hg(axis, model.phase_lobe2_g, rng);
    }
    const double selected_g = rng.uniform_open() < model.phase_lobe1_weight
                                  ? model.phase_lobe1_g
                                  : model.phase_lobe2_g;
    return sample_hg(axis, selected_g, rng);
}

HD Vec3 sample_lambertian_upward(CounterRng& rng) {
    const double radius = sqrt(rng.uniform_open());
    const double phi = 2.0 * kPi * rng.uniform_open();
    return make_vec(radius * cos(phi), radius * sin(phi),
                    sqrt(fmax(0.0, 1.0 - radius * radius)));
}

HD double boundary_distance(double vertical_tau, Vec3 direction, double slab_tau) {
    constexpr double parallel = 1.0e-15;
    if (direction.z < -parallel) return (slab_tau - vertical_tau) / (-direction.z);
    if (direction.z > parallel) return vertical_tau / direction.z;
    return 1.0e300;
}

HD double backward_path(uint64_t path_index, uint64_t seed, const Model& model,
                        bool* truncated) {
    *truncated = false;
    if (model.tau <= 0.0 || model.ssa <= 0.0) return 0.0;

    CounterRng rng(path_index, seed);
    double vertical_tau = 0.0;
    Vec3 direction = model.initial_backward;
    double weight = 1.0;
    double radiance_over_f0 = 0.0;

    for (int order = 0; order < model.max_scatters; ++order) {
        const double free_path = -log(rng.uniform_open());
        const double to_boundary = boundary_distance(vertical_tau, direction, model.tau);
        if (free_path >= to_boundary) return radiance_over_f0;

        vertical_tau -= direction.z * free_path;
        vertical_tau = fmin(model.tau, fmax(0.0, vertical_tau));

        const double scatter_cosine = dot(model.sun_direction, -direction);
        const double direct_transmittance = exp(-vertical_tau / model.mu0);
        radiance_over_f0 +=
            weight * model.ssa * hg_phase(scatter_cosine, model.g) * direct_transmittance;

        weight *= model.ssa;
        if (weight < 1.0e-14) return radiance_over_f0;
        direction = sample_hg(direction, model.g, rng);
    }

    *truncated = true;
    return radiance_over_f0;
}

__global__ void backward_kernel(uint64_t sample_offset, uint64_t sample_count, uint64_t seed,
                                Model model, double* block_sums, double* block_sums_sq,
                                uint32_t* block_truncated) {
    __shared__ double sums[kThreads];
    __shared__ double sums_sq[kThreads];
    __shared__ uint32_t truncated_counts[kThreads];

    const uint32_t lane = threadIdx.x;
    const uint64_t local_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + lane;
    double value = 0.0;
    bool truncated = false;
    if (local_index < sample_count) {
        value = backward_path(sample_offset + local_index, seed, model, &truncated);
    }
    sums[lane] = value;
    sums_sq[lane] = value * value;
    truncated_counts[lane] = truncated ? 1U : 0U;
    __syncthreads();

    for (uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            sums_sq[lane] += sums_sq[lane + stride];
            truncated_counts[lane] += truncated_counts[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        block_sums[blockIdx.x] = sums[0];
        block_sums_sq[blockIdx.x] = sums_sq[0];
        block_truncated[blockIdx.x] = truncated_counts[0];
    }
}

HD double backward_path_stage2(uint64_t path_index, uint64_t seed,
                               const Stage2Model& model, bool* truncated) {
    *truncated = false;
    CounterRng rng(path_index, seed);
    double vertical_tau = 0.0;
    Vec3 direction = model.initial_backward;
    double weight = 1.0;
    double radiance_over_f0 = 0.0;
    int scatter_order = 0;

    while (scatter_order < model.max_scatters) {
        const double free_path = -log(rng.uniform_open());
        const double to_boundary = boundary_distance(vertical_tau, direction, model.tau);
        if (free_path >= to_boundary) {
            if (direction.z > 0.0) return radiance_over_f0;

            // A Lambertian boundary contributes its directly illuminated source
            // and then samples the diffuse incident field.  With albedo zero this
            // returns before consuming any new random numbers, preserving the
            // legacy black-boundary identity path.
            if (model.surface_albedo <= 0.0) return radiance_over_f0;
            const double direct_surface = model.surface_albedo * model.mu0 / kPi *
                                          exp(-model.tau / model.mu0);
            radiance_over_f0 += weight * direct_surface;
            weight *= model.surface_albedo;
            if (weight < 1.0e-14) return radiance_over_f0;
            vertical_tau = model.tau;
            direction = sample_lambertian_upward(rng);
            continue;
        }

        vertical_tau -= direction.z * free_path;
        vertical_tau = fmin(model.tau, fmax(0.0, vertical_tau));

        const double scatter_cosine = dot(model.sun_direction, -direction);
        const double direct_transmittance = exp(-vertical_tau / model.mu0);
        radiance_over_f0 += weight * model.ssa *
                            mixture_hg_phase(scatter_cosine, model) *
                            direct_transmittance;

        weight *= model.ssa;
        if (weight < 1.0e-14) return radiance_over_f0;
        direction = sample_mixture_hg(direction, model, rng);
        ++scatter_order;
    }

    *truncated = true;
    return radiance_over_f0;
}

__global__ void backward_kernel_stage2(uint64_t sample_offset, uint64_t sample_count,
                                       uint64_t seed, Stage2Model model,
                                       double* block_sums, double* block_sums_sq,
                                       uint32_t* block_truncated) {
    __shared__ double sums[kThreads];
    __shared__ double sums_sq[kThreads];
    __shared__ uint32_t truncated_counts[kThreads];

    const uint32_t lane = threadIdx.x;
    const uint64_t local_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + lane;
    double value = 0.0;
    bool truncated = false;
    if (local_index < sample_count) {
        value = backward_path_stage2(sample_offset + local_index, seed, model, &truncated);
    }
    sums[lane] = value;
    sums_sq[lane] = value * value;
    truncated_counts[lane] = truncated ? 1U : 0U;
    __syncthreads();

    for (uint32_t stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            sums_sq[lane] += sums_sq[lane + stride];
            truncated_counts[lane] += truncated_counts[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        block_sums[blockIdx.x] = sums[0];
        block_sums_sq[blockIdx.x] = sums_sq[0];
        block_truncated[blockIdx.x] = truncated_counts[0];
    }
}

struct RawStats {
    long double sum = 0.0L;
    long double sum_sq = 0.0L;
    uint64_t truncated = 0;
};

struct Result {
    Parameters parameters;
    std::string backend;
    std::string device;
    RawStats raw;
    double mean_i_over_f0 = 0.0;
    double sample_variance_i_over_f0 = 0.0;
    double standard_error_i_over_f0 = 0.0;
    double brf = 0.0;
    double standard_error_brf = 0.0;
    double ci95_low_brf = 0.0;
    double ci95_high_brf = 0.0;
    double truncated_fraction = 0.0;
    double elapsed_ms = 0.0;
    double kernel_ms = 0.0;
    double max_batch_ms = 0.0;
    uint64_t batches = 0;
};

Result finalize_result(const Parameters& p, const std::string& backend,
                       const std::string& device, const RawStats& raw, double elapsed_ms,
                       double kernel_ms, double max_batch_ms, uint64_t batches) {
    Result out;
    out.parameters = p;
    out.backend = backend;
    out.device = device;
    out.raw = raw;
    out.elapsed_ms = elapsed_ms;
    out.kernel_ms = kernel_ms;
    out.max_batch_ms = max_batch_ms;
    out.batches = batches;

    const long double n = static_cast<long double>(p.samples);
    const long double mean = raw.sum / n;
    long double variance = 0.0L;
    if (p.samples > 1) {
        variance = (raw.sum_sq - raw.sum * raw.sum / n) / (n - 1.0L);
        if (variance < 0.0L && variance > -1.0e-24L) variance = 0.0L;
    }
    const long double stderr_i = sqrt(variance / n);
    const double brf_scale = kPi / std::cos(p.sun_zenith_deg * kPi / 180.0);
    out.mean_i_over_f0 = static_cast<double>(mean);
    out.sample_variance_i_over_f0 = static_cast<double>(variance);
    out.standard_error_i_over_f0 = static_cast<double>(stderr_i);
    out.brf = brf_scale * out.mean_i_over_f0;
    out.standard_error_brf = brf_scale * out.standard_error_i_over_f0;
    out.ci95_low_brf = out.brf - 1.96 * out.standard_error_brf;
    out.ci95_high_brf = out.brf + 1.96 * out.standard_error_brf;
    out.truncated_fraction = static_cast<double>(raw.truncated) / static_cast<double>(p.samples);
    return out;
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << operation << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

std::string cuda_device_name() {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    std::ostringstream name;
    name << properties.name << " (sm_" << properties.major << properties.minor << ")";
    return name.str();
}

Result run_gpu(const Parameters& p) {
    const Model model = make_model(p);
    const Stage2Model stage2_model = make_stage2_model(p);
    const uint64_t largest_batch = std::min(p.samples, p.batch_samples);
    const uint64_t largest_blocks = (largest_batch + kThreads - 1) / kThreads;
    if (largest_blocks > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("batch is too large for a CUDA grid");
    }

    double* device_sums = nullptr;
    double* device_sums_sq = nullptr;
    uint32_t* device_truncated = nullptr;
    cuda_check(cudaMalloc(&device_sums, largest_blocks * sizeof(double)), "cudaMalloc sums");
    try {
        cuda_check(cudaMalloc(&device_sums_sq, largest_blocks * sizeof(double)),
                   "cudaMalloc sums_sq");
        cuda_check(cudaMalloc(&device_truncated, largest_blocks * sizeof(uint32_t)),
                   "cudaMalloc truncated");
    } catch (...) {
        cudaFree(device_sums);
        cudaFree(device_sums_sq);
        throw;
    }

    cudaEvent_t event_start{};
    cudaEvent_t event_stop{};
    cuda_check(cudaEventCreate(&event_start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&event_stop), "cudaEventCreate stop");

    std::vector<double> host_sums(largest_blocks);
    std::vector<double> host_sums_sq(largest_blocks);
    std::vector<uint32_t> host_truncated(largest_blocks);
    RawStats raw;
    double total_kernel_ms = 0.0;
    double max_batch_ms = 0.0;
    uint64_t batches = 0;
    const auto wall_start = std::chrono::steady_clock::now();

    try {
        for (uint64_t offset = 0; offset < p.samples; offset += p.batch_samples) {
            const uint64_t count = std::min(p.batch_samples, p.samples - offset);
            const uint64_t blocks = (count + kThreads - 1) / kThreads;
            cuda_check(cudaEventRecord(event_start), "cudaEventRecord start");
            if (p.stage2_requested) {
                backward_kernel_stage2<<<static_cast<unsigned int>(blocks), kThreads>>>(
                    offset, count, p.seed, stage2_model, device_sums, device_sums_sq,
                    device_truncated);
                cuda_check(cudaGetLastError(), "backward_kernel_stage2 launch");
            } else {
                backward_kernel<<<static_cast<unsigned int>(blocks), kThreads>>>(
                    offset, count, p.seed, model, device_sums, device_sums_sq,
                    device_truncated);
                cuda_check(cudaGetLastError(), "backward_kernel launch");
            }
            cuda_check(cudaEventRecord(event_stop), "cudaEventRecord stop");
            cuda_check(cudaEventSynchronize(event_stop), "backward_kernel synchronize");
            float batch_ms = 0.0F;
            cuda_check(cudaEventElapsedTime(&batch_ms, event_start, event_stop),
                       "cudaEventElapsedTime");
            total_kernel_ms += batch_ms;
            max_batch_ms = std::max(max_batch_ms, static_cast<double>(batch_ms));

            cuda_check(cudaMemcpy(host_sums.data(), device_sums, blocks * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy sums");
            cuda_check(cudaMemcpy(host_sums_sq.data(), device_sums_sq, blocks * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy sums_sq");
            cuda_check(cudaMemcpy(host_truncated.data(), device_truncated,
                                  blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                       "cudaMemcpy truncated");
            for (uint64_t block = 0; block < blocks; ++block) {
                raw.sum += static_cast<long double>(host_sums[block]);
                raw.sum_sq += static_cast<long double>(host_sums_sq[block]);
                raw.truncated += host_truncated[block];
            }
            ++batches;
        }
    } catch (...) {
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_stop);
        cudaFree(device_sums);
        cudaFree(device_sums_sq);
        cudaFree(device_truncated);
        throw;
    }

    const auto wall_stop = std::chrono::steady_clock::now();
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
    cudaFree(device_sums);
    cudaFree(device_sums_sq);
    cudaFree(device_truncated);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    return finalize_result(p, "gpu", cuda_device_name(), raw, elapsed_ms, total_kernel_ms,
                           max_batch_ms, batches);
}

Result run_cpu(const Parameters& p) {
    const Model model = make_model(p);
    const Stage2Model stage2_model = make_stage2_model(p);
    RawStats raw;
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t sample = 0; sample < p.samples; ++sample) {
        bool truncated = false;
        const double value = p.stage2_requested
                                 ? backward_path_stage2(sample, p.seed, stage2_model, &truncated)
                                 : backward_path(sample, p.seed, model, &truncated);
        raw.sum += static_cast<long double>(value);
        raw.sum_sq += static_cast<long double>(value) * static_cast<long double>(value);
        raw.truncated += truncated ? 1U : 0U;
    }
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return finalize_result(p, "cpu", "host scalar reference", raw, elapsed_ms, 0.0, 0.0, 1);
}

double analytic_single_scatter_brf(const Parameters& p) {
    const Model model = make_model(p);
    if (p.tau <= 0.0 || p.ssa <= 0.0) return 0.0;
    const double scatter_cosine = dot(model.sun_direction, -model.initial_backward);
    const double attenuation =
        1.0 - std::exp(-p.tau * (1.0 / model.mu0 + 1.0 / model.muv));
    return kPi * p.ssa * hg_phase(scatter_cosine, p.g) * attenuation /
           (model.mu0 + model.muv);
}

double analytic_stage2_single_scatter_brf(const Parameters& p) {
    const Stage2Model model = make_stage2_model(p);
    if (p.tau <= 0.0 || p.ssa <= 0.0) return 0.0;
    const double scatter_cosine = dot(model.sun_direction, -model.initial_backward);
    const double attenuation =
        1.0 - std::exp(-p.tau * (1.0 / model.mu0 + 1.0 / model.muv));
    return kPi * p.ssa * mixture_hg_phase(scatter_cosine, model) * attenuation /
           (model.mu0 + model.muv);
}

struct EnergyResult {
    uint64_t reflected = 0;
    uint64_t transmitted = 0;
    uint64_t absorbed = 0;
    uint64_t truncated = 0;
    uint64_t samples = 0;
    double reflected_fraction = 0.0;
    double transmitted_fraction = 0.0;
    double absorbed_fraction = 0.0;
    double truncated_fraction = 0.0;
    double closure = 0.0;
};

EnergyResult run_forward_energy_check(const Parameters& p) {
    const Model model = make_model(p);
    EnergyResult out;
    out.samples = p.samples;

    for (uint64_t sample = 0; sample < p.samples; ++sample) {
        CounterRng rng(sample, p.seed ^ 0x454E455247590001ULL);
        double vertical_tau = 0.0;
        Vec3 direction = model.sun_direction;
        bool classified = false;
        for (int order = 0; order < p.max_scatters; ++order) {
            const double free_path = -std::log(rng.uniform_open());
            const double to_boundary = boundary_distance(vertical_tau, direction, p.tau);
            if (free_path >= to_boundary) {
                if (direction.z > 0.0) {
                    ++out.reflected;
                } else {
                    ++out.transmitted;
                }
                classified = true;
                break;
            }
            vertical_tau -= direction.z * free_path;
            vertical_tau = std::min(p.tau, std::max(0.0, vertical_tau));
            if (rng.uniform_open() > p.ssa) {
                ++out.absorbed;
                classified = true;
                break;
            }
            direction = sample_hg(direction, p.g, rng);
        }
        if (!classified) ++out.truncated;
    }

    const double inv = 1.0 / static_cast<double>(p.samples);
    out.reflected_fraction = static_cast<double>(out.reflected) * inv;
    out.transmitted_fraction = static_cast<double>(out.transmitted) * inv;
    out.absorbed_fraction = static_cast<double>(out.absorbed) * inv;
    out.truncated_fraction = static_cast<double>(out.truncated) * inv;
    out.closure = out.reflected_fraction + out.transmitted_fraction + out.absorbed_fraction +
                  out.truncated_fraction;
    return out;
}

struct Stage2EnergyResult {
    uint64_t reflected = 0;
    uint64_t transmitted = 0;
    uint64_t absorbed = 0;
    uint64_t truncated = 0;
    uint64_t samples = 0;
    std::array<uint64_t, kStage2DepthBins> collision_counts{};
    std::array<uint64_t, kStage2DepthBins> scattering_source_counts{};
    std::array<uint64_t, kStage2DepthBins> absorption_counts{};
    double reflected_fraction = 0.0;
    double transmitted_fraction = 0.0;
    double absorbed_fraction = 0.0;
    double truncated_fraction = 0.0;
    double closure = 0.0;
    double elapsed_ms = 0.0;
    double kernel_ms = 0.0;
    double max_batch_ms = 0.0;
    uint64_t batches = 0;
};

HD int stage2_depth_bin(double vertical_tau, double slab_tau, int depth_bins) {
    if (depth_bins <= 0 || slab_tau <= 0.0) return -1;
    int bin = static_cast<int>(vertical_tau / slab_tau * static_cast<double>(depth_bins));
    if (bin < 0) bin = 0;
    if (bin >= depth_bins) bin = depth_bins - 1;
    return bin;
}

__global__ void forward_kernel_stage2(
    uint64_t sample_offset, uint64_t sample_count, uint64_t seed, Stage2Model model,
    int depth_bins, unsigned long long* outcome_counts,
    unsigned long long* depth_collision_counts,
    unsigned long long* depth_scattering_source_counts,
    unsigned long long* depth_absorption_counts) {
    __shared__ uint32_t shared_outcomes[4];
    __shared__ uint32_t shared_collisions[kStage2DepthBins];
    __shared__ uint32_t shared_scattering_sources[kStage2DepthBins];
    __shared__ uint32_t shared_absorptions[kStage2DepthBins];

    const uint32_t lane = threadIdx.x;
    if (lane < 4) shared_outcomes[lane] = 0;
    if (lane < kStage2DepthBins) {
        shared_collisions[lane] = 0;
        shared_scattering_sources[lane] = 0;
        shared_absorptions[lane] = 0;
    }
    __syncthreads();

    const uint64_t local_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + lane;
    if (local_index < sample_count) {
        CounterRng rng(sample_offset + local_index, seed ^ 0x454E455247590002ULL);
        double vertical_tau = 0.0;
        Vec3 direction = model.sun_direction;
        int scatter_order = 0;
        int outcome = 3;

        while (scatter_order < model.max_scatters) {
            const double free_path = -log(rng.uniform_open());
            const double to_boundary = boundary_distance(vertical_tau, direction, model.tau);
            if (free_path >= to_boundary) {
                if (direction.z > 0.0) {
                    outcome = 0;
                    break;
                }

                bool reflect = false;
                if (model.surface_albedo >= 1.0) {
                    reflect = true;
                } else if (model.surface_albedo > 0.0) {
                    reflect = rng.uniform_open() < model.surface_albedo;
                }
                if (!reflect) {
                    // T is lower-boundary loss.  For an opaque physical surface
                    // it may equivalently be interpreted as surface absorption.
                    outcome = 1;
                    break;
                }
                vertical_tau = model.tau;
                direction = sample_lambertian_upward(rng);
                continue;
            }

            vertical_tau -= direction.z * free_path;
            vertical_tau = fmin(model.tau, fmax(0.0, vertical_tau));
            const int bin = stage2_depth_bin(vertical_tau, model.tau, depth_bins);
            if (bin >= 0) atomicAdd(&shared_collisions[bin], 1U);

            if (rng.uniform_open() > model.ssa) {
                if (bin >= 0) atomicAdd(&shared_absorptions[bin], 1U);
                outcome = 2;
                break;
            }
            if (bin >= 0) atomicAdd(&shared_scattering_sources[bin], 1U);
            direction = sample_mixture_hg(direction, model, rng);
            ++scatter_order;
        }
        atomicAdd(&shared_outcomes[outcome], 1U);
    }
    __syncthreads();

    if (lane < 4) {
        atomicAdd(&outcome_counts[lane],
                  static_cast<unsigned long long>(shared_outcomes[lane]));
    }
    if (lane < static_cast<uint32_t>(depth_bins)) {
        atomicAdd(&depth_collision_counts[lane],
                  static_cast<unsigned long long>(shared_collisions[lane]));
        atomicAdd(&depth_scattering_source_counts[lane],
                  static_cast<unsigned long long>(shared_scattering_sources[lane]));
        atomicAdd(&depth_absorption_counts[lane],
                  static_cast<unsigned long long>(shared_absorptions[lane]));
    }
}

void finalize_stage2_energy(Stage2EnergyResult& out) {
    const double inv = 1.0 / static_cast<double>(out.samples);
    out.reflected_fraction = static_cast<double>(out.reflected) * inv;
    out.transmitted_fraction = static_cast<double>(out.transmitted) * inv;
    out.absorbed_fraction = static_cast<double>(out.absorbed) * inv;
    out.truncated_fraction = static_cast<double>(out.truncated) * inv;
    out.closure = out.reflected_fraction + out.transmitted_fraction +
                  out.absorbed_fraction + out.truncated_fraction;
}

Stage2EnergyResult run_forward_stage2_cpu(const Parameters& p) {
    const Stage2Model model = make_stage2_model(p);
    Stage2EnergyResult out;
    out.samples = p.samples;
    out.batches = 1;
    const auto start = std::chrono::steady_clock::now();

    for (uint64_t sample = 0; sample < p.samples; ++sample) {
        CounterRng rng(sample, p.seed ^ 0x454E455247590002ULL);
        double vertical_tau = 0.0;
        Vec3 direction = model.sun_direction;
        int scatter_order = 0;
        bool classified = false;
        while (scatter_order < model.max_scatters) {
            const double free_path = -std::log(rng.uniform_open());
            const double to_boundary = boundary_distance(vertical_tau, direction, p.tau);
            if (free_path >= to_boundary) {
                if (direction.z > 0.0) {
                    ++out.reflected;
                    classified = true;
                    break;
                }
                bool reflect = false;
                if (p.surface_albedo >= 1.0) {
                    reflect = true;
                } else if (p.surface_albedo > 0.0) {
                    reflect = rng.uniform_open() < p.surface_albedo;
                }
                if (!reflect) {
                    ++out.transmitted;
                    classified = true;
                    break;
                }
                vertical_tau = p.tau;
                direction = sample_lambertian_upward(rng);
                continue;
            }

            vertical_tau -= direction.z * free_path;
            vertical_tau = std::min(p.tau, std::max(0.0, vertical_tau));
            const int bin = stage2_depth_bin(vertical_tau, p.tau, p.depth_bins);
            if (bin >= 0) ++out.collision_counts[bin];
            if (rng.uniform_open() > p.ssa) {
                if (bin >= 0) ++out.absorption_counts[bin];
                ++out.absorbed;
                classified = true;
                break;
            }
            if (bin >= 0) ++out.scattering_source_counts[bin];
            direction = sample_mixture_hg(direction, model, rng);
            ++scatter_order;
        }
        if (!classified) ++out.truncated;
    }

    const auto stop = std::chrono::steady_clock::now();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    finalize_stage2_energy(out);
    return out;
}

Stage2EnergyResult run_forward_stage2_gpu(const Parameters& p) {
    const Stage2Model model = make_stage2_model(p);
    Stage2EnergyResult out;
    out.samples = p.samples;

    unsigned long long* device_outcomes = nullptr;
    unsigned long long* device_collisions = nullptr;
    unsigned long long* device_scattering_sources = nullptr;
    unsigned long long* device_absorptions = nullptr;
    cuda_check(cudaMalloc(&device_outcomes, 4 * sizeof(unsigned long long)),
               "cudaMalloc stage2 outcomes");
    try {
        cuda_check(cudaMalloc(&device_collisions,
                              kStage2DepthBins * sizeof(unsigned long long)),
                   "cudaMalloc stage2 collisions");
        cuda_check(cudaMalloc(&device_scattering_sources,
                              kStage2DepthBins * sizeof(unsigned long long)),
                   "cudaMalloc stage2 scattering sources");
        cuda_check(cudaMalloc(&device_absorptions,
                              kStage2DepthBins * sizeof(unsigned long long)),
                   "cudaMalloc stage2 absorptions");
        cuda_check(cudaMemset(device_outcomes, 0, 4 * sizeof(unsigned long long)),
                   "cudaMemset stage2 outcomes");
        cuda_check(cudaMemset(device_collisions, 0,
                              kStage2DepthBins * sizeof(unsigned long long)),
                   "cudaMemset stage2 collisions");
        cuda_check(cudaMemset(device_scattering_sources, 0,
                              kStage2DepthBins * sizeof(unsigned long long)),
                   "cudaMemset stage2 scattering sources");
        cuda_check(cudaMemset(device_absorptions, 0,
                              kStage2DepthBins * sizeof(unsigned long long)),
                   "cudaMemset stage2 absorptions");
    } catch (...) {
        cudaFree(device_outcomes);
        cudaFree(device_collisions);
        cudaFree(device_scattering_sources);
        cudaFree(device_absorptions);
        throw;
    }

    cudaEvent_t event_start{};
    cudaEvent_t event_stop{};
    cuda_check(cudaEventCreate(&event_start), "cudaEventCreate stage2 forward start");
    cuda_check(cudaEventCreate(&event_stop), "cudaEventCreate stage2 forward stop");
    const auto wall_start = std::chrono::steady_clock::now();

    try {
        for (uint64_t offset = 0; offset < p.samples; offset += p.batch_samples) {
            const uint64_t count = std::min(p.batch_samples, p.samples - offset);
            const uint64_t blocks = (count + kThreads - 1) / kThreads;
            if (blocks > static_cast<uint64_t>(std::numeric_limits<unsigned int>::max())) {
                throw std::runtime_error("batch is too large for a CUDA grid");
            }
            cuda_check(cudaEventRecord(event_start), "cudaEventRecord stage2 forward start");
            forward_kernel_stage2<<<static_cast<unsigned int>(blocks), kThreads>>>(
                offset, count, p.seed, model, p.depth_bins, device_outcomes,
                device_collisions, device_scattering_sources, device_absorptions);
            cuda_check(cudaGetLastError(), "forward_kernel_stage2 launch");
            cuda_check(cudaEventRecord(event_stop), "cudaEventRecord stage2 forward stop");
            cuda_check(cudaEventSynchronize(event_stop),
                       "forward_kernel_stage2 synchronize");
            float batch_ms = 0.0F;
            cuda_check(cudaEventElapsedTime(&batch_ms, event_start, event_stop),
                       "cudaEventElapsedTime stage2 forward");
            out.kernel_ms += batch_ms;
            out.max_batch_ms = std::max(out.max_batch_ms, static_cast<double>(batch_ms));
            ++out.batches;
        }

        std::array<unsigned long long, 4> outcomes{};
        std::array<unsigned long long, kStage2DepthBins> collisions{};
        std::array<unsigned long long, kStage2DepthBins> scattering_sources{};
        std::array<unsigned long long, kStage2DepthBins> absorptions{};
        cuda_check(cudaMemcpy(outcomes.data(), device_outcomes,
                              outcomes.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy stage2 outcomes");
        cuda_check(cudaMemcpy(collisions.data(), device_collisions,
                              collisions.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy stage2 collisions");
        cuda_check(cudaMemcpy(scattering_sources.data(), device_scattering_sources,
                              scattering_sources.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy stage2 scattering sources");
        cuda_check(cudaMemcpy(absorptions.data(), device_absorptions,
                              absorptions.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy stage2 absorptions");
        out.reflected = outcomes[0];
        out.transmitted = outcomes[1];
        out.absorbed = outcomes[2];
        out.truncated = outcomes[3];
        for (int bin = 0; bin < kStage2DepthBins; ++bin) {
            out.collision_counts[bin] = collisions[bin];
            out.scattering_source_counts[bin] = scattering_sources[bin];
            out.absorption_counts[bin] = absorptions[bin];
        }
    } catch (...) {
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_stop);
        cudaFree(device_outcomes);
        cudaFree(device_collisions);
        cudaFree(device_scattering_sources);
        cudaFree(device_absorptions);
        throw;
    }

    const auto wall_stop = std::chrono::steady_clock::now();
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
    cudaFree(device_outcomes);
    cudaFree(device_collisions);
    cudaFree(device_scattering_sources);
    cudaFree(device_absorptions);
    out.elapsed_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    finalize_stage2_energy(out);
    return out;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

void write_result_json(std::ostream& out, const Result& r) {
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema\": \"simsat.cuda-cloud-oracle.result.v1\",\n";
    out << "  \"case\": \"" << json_escape(r.parameters.case_name) << "\",\n";
    out << "  \"backend\": \"" << r.backend << "\",\n";
    out << "  \"device\": \"" << json_escape(r.device) << "\",\n";
    out << "  \"tau\": " << r.parameters.tau << ",\n";
    out << "  \"ssa\": " << r.parameters.ssa << ",\n";
    out << "  \"hg_g\": " << r.parameters.g << ",\n";
    out << "  \"sun_zenith_deg\": " << r.parameters.sun_zenith_deg << ",\n";
    out << "  \"view_zenith_deg\": " << r.parameters.view_zenith_deg << ",\n";
    out << "  \"relative_azimuth_deg\": " << r.parameters.relative_azimuth_deg << ",\n";
    out << "  \"samples\": " << r.parameters.samples << ",\n";
    out << "  \"seed\": " << r.parameters.seed << ",\n";
    out << "  \"max_scatters\": " << r.parameters.max_scatters << ",\n";
    out << "  \"batch_samples\": " << r.parameters.batch_samples << ",\n";
    out << "  \"batches\": " << r.batches << ",\n";
    out << "  \"mean_i_over_f0\": " << r.mean_i_over_f0 << ",\n";
    out << "  \"sample_variance_i_over_f0\": " << r.sample_variance_i_over_f0 << ",\n";
    out << "  \"standard_error_i_over_f0\": " << r.standard_error_i_over_f0 << ",\n";
    out << "  \"brf\": " << r.brf << ",\n";
    out << "  \"standard_error_brf\": " << r.standard_error_brf << ",\n";
    out << "  \"ci95_low_brf\": " << r.ci95_low_brf << ",\n";
    out << "  \"ci95_high_brf\": " << r.ci95_high_brf << ",\n";
    out << "  \"truncated_paths\": " << r.raw.truncated << ",\n";
    out << "  \"truncated_fraction\": " << r.truncated_fraction << ",\n";
    out << "  \"elapsed_ms\": " << r.elapsed_ms << ",\n";
    out << "  \"kernel_ms\": " << r.kernel_ms << ",\n";
    out << "  \"max_batch_ms\": " << r.max_batch_ms << "\n";
    out << "}\n";
}

void write_stage2_result_json(std::ostream& out, const Result& r,
                              const Stage2EnergyResult* energy) {
    const Parameters& p = r.parameters;
    const double phase_first_moment =
        p.phase_lobe1_weight * p.g + (1.0 - p.phase_lobe1_weight) * p.phase_lobe2_g;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema\": \"simsat.cuda-cloud-oracle.result.v2\",\n";
    out << "  \"schema_version\": 2,\n";
    out << "  \"capability_set\": \"stage2-reference-v1\",\n";
    out << "  \"capabilities\": {\n";
    out << "    \"lambertian_lower_boundary\": 1,\n";
    out << "    \"mixture_hg_sampling\": 1,\n";
    out << "    \"matched_forward_rta\": 1,\n";
    out << "    \"depth_binned_collision_source\": 1\n";
    out << "  },\n";
    out << "  \"request_contract\": \"simsat.cloud-closure-fit.stage2-request-grid.v1\",\n";
    out << "  \"case\": \"" << json_escape(p.case_name) << "\",\n";
    out << "  \"backend\": \"" << r.backend << "\",\n";
    out << "  \"device\": \"" << json_escape(r.device) << "\",\n";
    out << "  \"tau\": " << p.tau << ",\n";
    out << "  \"ssa\": " << p.ssa << ",\n";
    out << "  \"phase_model\": \"" << json_escape(p.phase_model) << "\",\n";
    out << "  \"phase_lobe1_g\": " << p.g << ",\n";
    out << "  \"phase_lobe2_g\": " << p.phase_lobe2_g << ",\n";
    out << "  \"phase_lobe1_weight\": " << p.phase_lobe1_weight << ",\n";
    out << "  \"phase_first_moment\": " << phase_first_moment << ",\n";
    out << "  \"lower_boundary\": \"" << json_escape(p.lower_boundary) << "\",\n";
    out << "  \"surface_albedo\": " << p.surface_albedo << ",\n";
    out << "  \"sun_zenith_deg\": " << p.sun_zenith_deg << ",\n";
    out << "  \"view_zenith_deg\": " << p.view_zenith_deg << ",\n";
    out << "  \"relative_azimuth_deg\": " << p.relative_azimuth_deg << ",\n";
    out << "  \"samples\": " << p.samples << ",\n";
    out << "  \"seed\": " << p.seed << ",\n";
    out << "  \"max_scatters\": " << p.max_scatters << ",\n";
    out << "  \"batch_samples\": " << p.batch_samples << ",\n";
    out << "  \"depth_bins\": " << p.depth_bins << ",\n";
    out << "  \"report_forward_flux\": "
        << (p.report_forward_flux ? "true" : "false") << ",\n";
    out << "  \"directional\": {\n";
    out << "    \"observable\": \"BRF = pi I / (F0 mu0)\",\n";
    out << "    \"batches\": " << r.batches << ",\n";
    out << "    \"mean_i_over_f0\": " << r.mean_i_over_f0 << ",\n";
    out << "    \"sample_variance_i_over_f0\": " << r.sample_variance_i_over_f0 << ",\n";
    out << "    \"standard_error_i_over_f0\": " << r.standard_error_i_over_f0 << ",\n";
    out << "    \"brf\": " << r.brf << ",\n";
    out << "    \"standard_error_brf\": " << r.standard_error_brf << ",\n";
    out << "    \"ci95_low_brf\": " << r.ci95_low_brf << ",\n";
    out << "    \"ci95_high_brf\": " << r.ci95_high_brf << ",\n";
    out << "    \"truncated_paths\": " << r.raw.truncated << ",\n";
    out << "    \"truncated_fraction\": " << r.truncated_fraction << ",\n";
    out << "    \"elapsed_ms\": " << r.elapsed_ms << ",\n";
    out << "    \"kernel_ms\": " << r.kernel_ms << ",\n";
    out << "    \"max_batch_ms\": " << r.max_batch_ms << "\n";
    out << "  },\n";

    if (energy == nullptr) {
        out << "  \"forward_flux\": null,\n";
        out << "  \"collision_source\": null\n";
        out << "}\n";
        return;
    }

    out << "  \"forward_flux\": {\n";
    out << "    \"normalization\": \"unit incident horizontal flux\",\n";
    out << "    \"transmitted_semantics\": \"lower-boundary loss; for an opaque surface this is surface absorption\",\n";
    out << "    \"absorbed_semantics\": \"volume absorption\",\n";
    out << "    \"rng_stream\": \"seed xor 0x454e455247590002\",\n";
    out << "    \"samples\": " << energy->samples << ",\n";
    out << "    \"reflected_paths\": " << energy->reflected << ",\n";
    out << "    \"transmitted_paths\": " << energy->transmitted << ",\n";
    out << "    \"absorbed_paths\": " << energy->absorbed << ",\n";
    out << "    \"truncated_paths\": " << energy->truncated << ",\n";
    out << "    \"R\": " << energy->reflected_fraction << ",\n";
    out << "    \"T\": " << energy->transmitted_fraction << ",\n";
    out << "    \"A\": " << energy->absorbed_fraction << ",\n";
    out << "    \"truncated_fraction\": " << energy->truncated_fraction << ",\n";
    out << "    \"closure\": " << energy->closure << ",\n";
    out << "    \"batches\": " << energy->batches << ",\n";
    out << "    \"elapsed_ms\": " << energy->elapsed_ms << ",\n";
    out << "    \"kernel_ms\": " << energy->kernel_ms << ",\n";
    out << "    \"max_batch_ms\": " << energy->max_batch_ms << "\n";
    out << "  },\n";
    out << "  \"collision_source\": {\n";
    out << "    \"coordinate\": \"fractional vertical optical depth, top=0, lower boundary=1\",\n";
    out << "    \"definition\": \"analog scattering collisions per incident path per unit fractional depth, integrated over angle\",\n";
    out << "    \"bin_count\": " << p.depth_bins << ",\n";
    out << "    \"bins\": [";
    if (p.depth_bins > 0) out << '\n';
    const double inv_samples = 1.0 / static_cast<double>(energy->samples);
    for (int bin = 0; bin < p.depth_bins; ++bin) {
        const double lower = static_cast<double>(bin) / static_cast<double>(p.depth_bins);
        const double upper = static_cast<double>(bin + 1) / static_cast<double>(p.depth_bins);
        const double source_per_path =
            static_cast<double>(energy->scattering_source_counts[bin]) * inv_samples;
        const double source_density = source_per_path * static_cast<double>(p.depth_bins);
        out << "      {\"index\": " << bin << ", \"fractional_depth_min\": " << lower
            << ", \"fractional_depth_max\": " << upper
            << ", \"collision_count\": " << energy->collision_counts[bin]
            << ", \"scattering_source_count\": "
            << energy->scattering_source_counts[bin]
            << ", \"absorption_count\": " << energy->absorption_counts[bin]
            << ", \"scattering_source_per_incident_path\": " << source_per_path
            << ", \"scattering_source_density\": " << source_density << '}';
        out << (bin + 1 == p.depth_bins ? "\n" : ",\n");
    }
    out << "    ]\n";
    out << "  }\n";
    out << "}\n";
}

void write_csv_header(std::ostream& out) {
    out << "case,backend,device,tau,ssa,hg_g,sun_zenith_deg,view_zenith_deg,"
           "relative_azimuth_deg,samples,seed,max_scatters,batch_samples,batches,"
           "mean_i_over_f0,sample_variance_i_over_f0,standard_error_i_over_f0,brf,"
           "standard_error_brf,ci95_low_brf,ci95_high_brf,truncated_paths,"
           "truncated_fraction,elapsed_ms,kernel_ms,max_batch_ms\n";
}

std::string csv_quote(const std::string& value) {
    std::string escaped;
    for (const char c : value) {
        if (c == '"') escaped += '"';
        escaped += c;
    }
    return '"' + escaped + '"';
}

void write_result_csv(std::ostream& out, const Result& r) {
    out << std::setprecision(17);
    out << csv_quote(r.parameters.case_name) << ',' << r.backend << ',' << csv_quote(r.device)
        << ',' << r.parameters.tau << ',' << r.parameters.ssa << ',' << r.parameters.g << ','
        << r.parameters.sun_zenith_deg << ',' << r.parameters.view_zenith_deg << ','
        << r.parameters.relative_azimuth_deg << ',' << r.parameters.samples << ','
        << r.parameters.seed << ',' << r.parameters.max_scatters << ','
        << r.parameters.batch_samples << ',' << r.batches << ',' << r.mean_i_over_f0 << ','
        << r.sample_variance_i_over_f0 << ',' << r.standard_error_i_over_f0 << ',' << r.brf
        << ',' << r.standard_error_brf << ',' << r.ci95_low_brf << ',' << r.ci95_high_brf
        << ',' << r.raw.truncated << ',' << r.truncated_fraction << ',' << r.elapsed_ms << ','
        << r.kernel_ms << ',' << r.max_batch_ms << '\n';
}

struct TestResult {
    std::string name;
    bool passed;
    std::string detail;
};

void add_test(std::vector<TestResult>& tests, const std::string& name, bool passed,
              const std::string& detail) {
    tests.push_back({name, passed, detail});
    std::cerr << (passed ? "PASS " : "FAIL ") << name << ": " << detail << '\n';
}

std::string number_detail(const std::string& label, double value) {
    std::ostringstream out;
    out << std::setprecision(17) << label << '=' << value;
    return out.str();
}

bool run_self_tests(std::ostream& output) {
    std::vector<TestResult> tests;
    const auto suite_start = std::chrono::steady_clock::now();

    const U32x4 known = philox4x32_10({0, 0, 0, 0}, 0, 0);
    const bool philox_ok = known.x == 0x6627E8D5U && known.y == 0xE169C58DU &&
                           known.z == 0xBC57AC4CU && known.w == 0x9B00DBD8U;
    std::ostringstream philox_detail;
    philox_detail << std::hex << std::setfill('0') << std::setw(8) << known.x << ' '
                  << std::setw(8) << known.y << ' ' << std::setw(8) << known.z << ' '
                  << std::setw(8) << known.w;
    add_test(tests, "philox4x32_10_known_vector", philox_ok, philox_detail.str());

    Parameters zero;
    zero.case_name = "tau_zero";
    zero.tau = 0.0;
    zero.samples = 65536;
    zero.max_scatters = 16;
    const Result zero_cpu = run_cpu(zero);
    const Result zero_gpu = run_gpu(zero);
    add_test(tests, "tau_zero_cpu_gpu", zero_cpu.brf == 0.0 && zero_gpu.brf == 0.0,
             number_detail("cpu_brf", zero_cpu.brf) + ", " +
                 number_detail("gpu_brf", zero_gpu.brf));

    Parameters absorbing = zero;
    absorbing.case_name = "ssa_zero";
    absorbing.tau = 5.0;
    absorbing.ssa = 0.0;
    const Result absorbing_cpu = run_cpu(absorbing);
    const Result absorbing_gpu = run_gpu(absorbing);
    add_test(tests, "ssa_zero_cpu_gpu",
             absorbing_cpu.brf == 0.0 && absorbing_gpu.brf == 0.0,
             number_detail("cpu_brf", absorbing_cpu.brf) + ", " +
                 number_detail("gpu_brf", absorbing_gpu.brf));

    Parameters thin;
    thin.case_name = "thin_single_scatter";
    thin.tau = 0.05;
    thin.ssa = 0.999;
    thin.g = 0.85;
    thin.sun_zenith_deg = 30.0;
    thin.view_zenith_deg = 20.0;
    thin.relative_azimuth_deg = 60.0;
    thin.samples = 1048576;
    thin.max_scatters = 1;
    thin.batch_samples = 65536;
    const Result thin_cpu = run_cpu(thin);
    const Result thin_gpu = run_gpu(thin);
    const double thin_analytic = analytic_single_scatter_brf(thin);
    const double thin_cpu_delta = std::fabs(thin_cpu.brf - thin_analytic);
    const double thin_gpu_delta = std::fabs(thin_gpu.brf - thin_analytic);
    const double thin_cpu_tolerance = std::max(6.0 * thin_cpu.standard_error_brf, 5.0e-10);
    const double thin_gpu_tolerance = std::max(6.0 * thin_gpu.standard_error_brf, 5.0e-10);
    std::ostringstream thin_detail;
    thin_detail << std::setprecision(17) << "cpu=" << thin_cpu.brf << ", gpu=" << thin_gpu.brf
                << ", analytic=" << thin_analytic << ", cpu_abs_delta=" << thin_cpu_delta
                << ", gpu_abs_delta=" << thin_gpu_delta
                << ", cpu_six_sigma=" << thin_cpu_tolerance
                << ", gpu_six_sigma=" << thin_gpu_tolerance;
    add_test(tests, "optically_thin_single_scatter_analytic",
             thin_cpu_delta <= thin_cpu_tolerance && thin_gpu_delta <= thin_gpu_tolerance,
             thin_detail.str());

    Parameters agreement;
    agreement.case_name = "cpu_gpu_agreement";
    agreement.tau = 2.0;
    agreement.ssa = 0.995;
    agreement.g = 0.8;
    agreement.sun_zenith_deg = 45.0;
    agreement.view_zenith_deg = 35.0;
    agreement.relative_azimuth_deg = 120.0;
    agreement.samples = 262144;
    agreement.max_scatters = 96;
    agreement.batch_samples = 65536;
    const Result agreement_cpu = run_cpu(agreement);
    const Result agreement_gpu = run_gpu(agreement);
    const double agreement_delta = std::fabs(agreement_cpu.brf - agreement_gpu.brf);
    const double agreement_sigma = std::sqrt(
        agreement_cpu.standard_error_brf * agreement_cpu.standard_error_brf +
        agreement_gpu.standard_error_brf * agreement_gpu.standard_error_brf);
    const double agreement_tolerance = std::max(6.0 * agreement_sigma, 1.0e-11);
    std::ostringstream agreement_detail;
    agreement_detail << std::setprecision(17) << "cpu=" << agreement_cpu.brf
                     << ", gpu=" << agreement_gpu.brf << ", abs_delta=" << agreement_delta
                     << ", six_combined_sigma=" << agreement_tolerance;
    add_test(tests, "cpu_gpu_statistical_agreement", agreement_delta <= agreement_tolerance,
             agreement_detail.str());

    const Result repeat_a = run_gpu(agreement);
    const Result repeat_b = run_gpu(agreement);
    const bool repeat_ok = repeat_a.raw.sum == repeat_b.raw.sum &&
                           repeat_a.raw.sum_sq == repeat_b.raw.sum_sq &&
                           repeat_a.raw.truncated == repeat_b.raw.truncated &&
                           repeat_a.brf == repeat_b.brf;
    std::ostringstream repeat_detail;
    repeat_detail << std::setprecision(17) << "brf_a=" << repeat_a.brf
                  << ", brf_b=" << repeat_b.brf
                  << ", truncated=" << repeat_a.raw.truncated;
    add_test(tests, "gpu_exact_repeatability", repeat_ok, repeat_detail.str());

    Parameters stress;
    stress.case_name = "finite_stress";
    stress.tau = 40.0;
    stress.ssa = 0.9999;
    stress.g = 0.9;
    stress.sun_zenith_deg = 75.0;
    stress.view_zenith_deg = 70.0;
    stress.relative_azimuth_deg = 180.0;
    stress.samples = 262144;
    stress.max_scatters = 256;
    stress.batch_samples = 32768;
    const Result stress_gpu = run_gpu(stress);
    const double p_max = (1.0 + stress.g) /
                         (4.0 * kPi * (1.0 - stress.g) * (1.0 - stress.g));
    const double order_sum = stress.ssa == 1.0
                                 ? static_cast<double>(stress.max_scatters)
                                 : stress.ssa * (1.0 - std::pow(stress.ssa, stress.max_scatters)) /
                                       (1.0 - stress.ssa);
    const double pathwise_brf_bound = kPi / std::cos(stress.sun_zenith_deg * kPi / 180.0) *
                                      p_max * order_sum;
    const bool stress_ok = std::isfinite(stress_gpu.brf) && stress_gpu.brf >= 0.0 &&
                           stress_gpu.brf <= pathwise_brf_bound &&
                           stress_gpu.truncated_fraction >= 0.0 &&
                           stress_gpu.truncated_fraction <= 1.0;
    std::ostringstream stress_detail;
    stress_detail << std::setprecision(17) << "brf=" << stress_gpu.brf
                  << ", pathwise_bound=" << pathwise_brf_bound
                  << ", truncated_fraction=" << stress_gpu.truncated_fraction
                  << ", max_batch_ms=" << stress_gpu.max_batch_ms;
    add_test(tests, "finite_nonnegative_pathwise_bound", stress_ok, stress_detail.str());

    Parameters energy;
    energy.case_name = "forward_energy";
    energy.tau = 5.0;
    energy.ssa = 0.98;
    energy.g = 0.85;
    energy.sun_zenith_deg = 60.0;
    energy.samples = 262144;
    energy.max_scatters = 512;
    const EnergyResult energy_result = run_forward_energy_check(energy);
    const bool energy_ok = std::isfinite(energy_result.closure) &&
                           std::fabs(energy_result.closure - 1.0) <= 4.0e-15 &&
                           energy_result.reflected_fraction >= 0.0 &&
                           energy_result.transmitted_fraction >= 0.0 &&
                           energy_result.absorbed_fraction >= 0.0 &&
                           energy_result.truncated_fraction >= 0.0;
    std::ostringstream energy_detail;
    energy_detail << std::setprecision(17) << "R=" << energy_result.reflected_fraction
                  << ", T=" << energy_result.transmitted_fraction
                  << ", A=" << energy_result.absorbed_fraction
                  << ", truncated=" << energy_result.truncated_fraction
                  << ", closure=" << energy_result.closure;
    add_test(tests, "forward_analog_energy_closure", energy_ok, energy_detail.str());

    Parameters identity = agreement;
    identity.case_name = "stage2_single_hg_black_identity";
    identity.phase_model = "single-hg";
    identity.phase_lobe2_g = 0.0;
    identity.phase_lobe1_weight = 1.0;
    identity.surface_albedo = 0.0;
    identity.stage2_requested = true;
    const Result identity_cpu = run_cpu(identity);
    const Result identity_gpu = run_gpu(identity);
    const bool identity_ok = identity_cpu.raw.sum == agreement_cpu.raw.sum &&
                             identity_cpu.raw.sum_sq == agreement_cpu.raw.sum_sq &&
                             identity_cpu.raw.truncated == agreement_cpu.raw.truncated &&
                             identity_gpu.raw.sum == agreement_gpu.raw.sum &&
                             identity_gpu.raw.sum_sq == agreement_gpu.raw.sum_sq &&
                             identity_gpu.raw.truncated == agreement_gpu.raw.truncated;
    std::ostringstream identity_detail;
    identity_detail << std::setprecision(17) << "legacy_cpu=" << agreement_cpu.brf
                    << ", stage2_cpu=" << identity_cpu.brf
                    << ", legacy_gpu=" << agreement_gpu.brf
                    << ", stage2_gpu=" << identity_gpu.brf;
    add_test(tests, "stage2_single_hg_black_boundary_exact_identity", identity_ok,
             identity_detail.str());

    Parameters lambertian;
    lambertian.case_name = "stage2_lambertian_tau_zero";
    lambertian.tau = 0.0;
    lambertian.surface_albedo = 0.6;
    lambertian.samples = 65536;
    lambertian.max_scatters = 16;
    lambertian.batch_samples = 16384;
    lambertian.stage2_requested = true;
    const Result lambertian_cpu = run_cpu(lambertian);
    const Result lambertian_gpu = run_gpu(lambertian);
    // MSVC aliases long double to double, so summing the identical per-path
    // surface contribution accumulates about 9e-13 at this sample count.
    const double lambertian_tolerance = 2.0e-12;
    const bool lambertian_ok =
        std::fabs(lambertian_cpu.brf - lambertian.surface_albedo) <= lambertian_tolerance &&
        std::fabs(lambertian_gpu.brf - lambertian.surface_albedo) <= lambertian_tolerance;
    std::ostringstream lambertian_detail;
    lambertian_detail << std::setprecision(17) << "expected=" << lambertian.surface_albedo
                      << ", cpu=" << lambertian_cpu.brf
                      << ", gpu=" << lambertian_gpu.brf;
    add_test(tests, "stage2_lambertian_tau_zero_analytic", lambertian_ok,
             lambertian_detail.str());

    Parameters dual = thin;
    dual.case_name = "stage2_dual_hg_single_scatter";
    dual.phase_model = "mixture-hg";
    dual.g = 0.85;
    dual.phase_lobe2_g = -0.15;
    dual.phase_lobe1_weight = 0.9;
    dual.expected_phase_first_moment = 0.75;
    dual.surface_albedo = 0.0;
    dual.stage2_requested = true;
    const Result dual_cpu = run_cpu(dual);
    const Result dual_gpu = run_gpu(dual);
    const double dual_analytic = analytic_stage2_single_scatter_brf(dual);
    const double dual_cpu_delta = std::fabs(dual_cpu.brf - dual_analytic);
    const double dual_gpu_delta = std::fabs(dual_gpu.brf - dual_analytic);
    const double dual_cpu_tolerance = std::max(6.0 * dual_cpu.standard_error_brf, 5.0e-10);
    const double dual_gpu_tolerance = std::max(6.0 * dual_gpu.standard_error_brf, 5.0e-10);
    std::ostringstream dual_detail;
    dual_detail << std::setprecision(17) << "first_moment="
                << dual.phase_lobe1_weight * dual.g +
                       (1.0 - dual.phase_lobe1_weight) * dual.phase_lobe2_g
                << ", cpu=" << dual_cpu.brf << ", gpu=" << dual_gpu.brf
                << ", analytic=" << dual_analytic;
    add_test(tests, "stage2_dual_hg_matched_moment_single_scatter_analytic",
             dual_cpu_delta <= dual_cpu_tolerance && dual_gpu_delta <= dual_gpu_tolerance,
             dual_detail.str());

    Parameters stage2_energy = energy;
    stage2_energy.case_name = "stage2_forward_energy_depth_source";
    stage2_energy.tau = 3.0;
    stage2_energy.ssa = 0.97;
    stage2_energy.phase_model = "mixture-hg";
    stage2_energy.g = 0.85;
    stage2_energy.phase_lobe2_g = -0.15;
    stage2_energy.phase_lobe1_weight = 0.9;
    stage2_energy.expected_phase_first_moment = 0.75;
    stage2_energy.surface_albedo = 0.6;
    stage2_energy.samples = 131072;
    stage2_energy.max_scatters = 256;
    stage2_energy.batch_samples = 32768;
    stage2_energy.depth_bins = kStage2DepthBins;
    stage2_energy.report_forward_flux = true;
    stage2_energy.stage2_requested = true;
    const Stage2EnergyResult stage2_energy_cpu = run_forward_stage2_cpu(stage2_energy);
    const Stage2EnergyResult stage2_energy_gpu = run_forward_stage2_gpu(stage2_energy);
    bool bin_accounting_ok = true;
    uint64_t cpu_collision_total = 0;
    uint64_t gpu_collision_total = 0;
    for (int bin = 0; bin < kStage2DepthBins; ++bin) {
        bin_accounting_ok = bin_accounting_ok &&
            stage2_energy_cpu.collision_counts[bin] ==
                stage2_energy_cpu.scattering_source_counts[bin] +
                    stage2_energy_cpu.absorption_counts[bin] &&
            stage2_energy_gpu.collision_counts[bin] ==
                stage2_energy_gpu.scattering_source_counts[bin] +
                    stage2_energy_gpu.absorption_counts[bin];
        cpu_collision_total += stage2_energy_cpu.collision_counts[bin];
        gpu_collision_total += stage2_energy_gpu.collision_counts[bin];
    }
    const double max_rta_delta = std::max(
        std::fabs(stage2_energy_cpu.reflected_fraction -
                  stage2_energy_gpu.reflected_fraction),
        std::max(std::fabs(stage2_energy_cpu.transmitted_fraction -
                           stage2_energy_gpu.transmitted_fraction),
                 std::fabs(stage2_energy_cpu.absorbed_fraction -
                           stage2_energy_gpu.absorbed_fraction)));
    const bool stage2_energy_ok =
        std::fabs(stage2_energy_cpu.closure - 1.0) <= 4.0e-15 &&
        std::fabs(stage2_energy_gpu.closure - 1.0) <= 4.0e-15 &&
        max_rta_delta <= 0.01 && bin_accounting_ok && cpu_collision_total > 0 &&
        gpu_collision_total > 0;
    std::ostringstream stage2_energy_detail;
    stage2_energy_detail << std::setprecision(17)
                         << "cpu_RTA=" << stage2_energy_cpu.reflected_fraction << '/'
                         << stage2_energy_cpu.transmitted_fraction << '/'
                         << stage2_energy_cpu.absorbed_fraction << ", gpu_RTA="
                         << stage2_energy_gpu.reflected_fraction << '/'
                         << stage2_energy_gpu.transmitted_fraction << '/'
                         << stage2_energy_gpu.absorbed_fraction
                         << ", max_delta=" << max_rta_delta
                         << ", cpu_collisions=" << cpu_collision_total
                         << ", gpu_collisions=" << gpu_collision_total;
    add_test(tests, "stage2_matched_forward_rta_depth_bin_energy", stage2_energy_ok,
             stage2_energy_detail.str());

    const Stage2EnergyResult stage2_repeat_a = run_forward_stage2_gpu(stage2_energy);
    const Stage2EnergyResult stage2_repeat_b = run_forward_stage2_gpu(stage2_energy);
    const bool stage2_repeat_ok =
        stage2_repeat_a.reflected == stage2_repeat_b.reflected &&
        stage2_repeat_a.transmitted == stage2_repeat_b.transmitted &&
        stage2_repeat_a.absorbed == stage2_repeat_b.absorbed &&
        stage2_repeat_a.truncated == stage2_repeat_b.truncated &&
        stage2_repeat_a.collision_counts == stage2_repeat_b.collision_counts &&
        stage2_repeat_a.scattering_source_counts ==
            stage2_repeat_b.scattering_source_counts &&
        stage2_repeat_a.absorption_counts == stage2_repeat_b.absorption_counts;
    std::ostringstream stage2_repeat_detail;
    stage2_repeat_detail << "R/T/A/truncated=" << stage2_repeat_a.reflected << '/'
                         << stage2_repeat_a.transmitted << '/' << stage2_repeat_a.absorbed
                         << '/' << stage2_repeat_a.truncated;
    add_test(tests, "stage2_forward_flux_depth_source_exact_repeatability",
             stage2_repeat_ok, stage2_repeat_detail.str());

    const auto suite_stop = std::chrono::steady_clock::now();
    const double suite_ms =
        std::chrono::duration<double, std::milli>(suite_stop - suite_start).count();
    const bool all_passed =
        std::all_of(tests.begin(), tests.end(), [](const TestResult& test) { return test.passed; });

    output << std::setprecision(17);
    output << "{\n  \"schema\": \"simsat.cuda-cloud-oracle.self-test.v1\",\n";
    output << "  \"device\": \"" << json_escape(cuda_device_name()) << "\",\n";
    output << "  \"all_passed\": " << (all_passed ? "true" : "false") << ",\n";
    output << "  \"elapsed_ms\": " << suite_ms << ",\n";
    output << "  \"tests\": [\n";
    for (size_t i = 0; i < tests.size(); ++i) {
        const TestResult& test = tests[i];
        output << "    {\"name\": \"" << json_escape(test.name) << "\", \"passed\": "
               << (test.passed ? "true" : "false") << ", \"detail\": \""
               << json_escape(test.detail) << "\"}";
        output << (i + 1 == tests.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return all_passed;
}

std::vector<Parameters> make_sweep(uint64_t samples, uint64_t batch_samples, uint64_t seed) {
    auto make = [&](const std::string& name, double tau, double ssa, double g, double sun,
                    double view, double az, int max_scatters = 256) {
        Parameters p;
        p.case_name = name;
        p.tau = tau;
        p.ssa = ssa;
        p.g = g;
        p.sun_zenith_deg = sun;
        p.view_zenith_deg = view;
        p.relative_azimuth_deg = az;
        p.samples = samples;
        p.seed = seed;
        p.max_scatters = max_scatters;
        p.batch_samples = batch_samples;
        return p;
    };

    return {
        make("clear_tau_0", 0.0, 0.999, 0.85, 30.0, 0.0, 0.0, 32),
        make("hrrr_like_wispy_tail_probe_tau_0p15", 0.15, 0.999, 0.85, 30.0, 0.0, 0.0,
             64),
        make("thin_liquid_tau_0p5", 0.5, 0.999, 0.85, 30.0, 0.0, 0.0, 96),
        make("nssl_like_liquid_edge_probe_tau_2", 2.0, 0.999, 0.85, 30.0, 0.0, 0.0,
             192),
        make("liquid_mid_tau_8", 8.0, 0.999, 0.85, 30.0, 0.0, 0.0, 384),
        make("thick_liquid_tau_25", 25.0, 0.999, 0.85, 30.0, 0.0, 0.0, 512),
        make("thin_ice_proxy_tau_0p3", 0.3, 0.999, 0.75, 30.0, 0.0, 0.0, 96),
        make("nssl_like_ice_anvil_probe_tau_1", 1.0, 0.999, 0.75, 30.0, 0.0, 0.0, 160),
        make("ice_anvil_proxy_tau_3", 3.0, 0.999, 0.75, 30.0, 0.0, 0.0, 256),
        make("oblique_sun_liquid_tau_2", 2.0, 0.999, 0.85, 70.0, 0.0, 0.0, 256),
        make("oblique_view_liquid_tau_2", 2.0, 0.999, 0.85, 30.0, 70.0, 90.0, 256),
        make("azimuthal_lobe_liquid_tau_2", 2.0, 0.999, 0.85, 60.0, 60.0, 180.0, 256),
        make("absorbing_stress_tau_2", 2.0, 0.95, 0.85, 30.0, 0.0, 0.0, 192),
    };
}

uint64_t parse_u64(const std::string& text, const char* option) {
    size_t consumed = 0;
    const uint64_t value = std::stoull(text, &consumed, 0);
    if (consumed != text.size()) throw std::runtime_error(std::string("invalid ") + option);
    return value;
}

int parse_int(const std::string& text, const char* option) {
    size_t consumed = 0;
    const int value = std::stoi(text, &consumed, 0);
    if (consumed != text.size()) throw std::runtime_error(std::string("invalid ") + option);
    return value;
}

double parse_double(const std::string& text, const char* option) {
    size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size()) throw std::runtime_error(std::string("invalid ") + option);
    return value;
}

bool parse_bool(const std::string& text, const char* option) {
    if (text == "true" || text == "1") return true;
    if (text == "false" || text == "0") return false;
    throw std::runtime_error(std::string("invalid ") + option + ": expected true or false");
}

void validate(const Parameters& p) {
    if (!(std::isfinite(p.tau) && p.tau >= 0.0)) throw std::runtime_error("tau must be finite and >= 0");
    if (!(std::isfinite(p.ssa) && p.ssa >= 0.0 && p.ssa <= 1.0))
        throw std::runtime_error("ssa must be in [0, 1]");
    if (!(std::isfinite(p.g) && p.g > -1.0 && p.g < 1.0))
        throw std::runtime_error("g must be in (-1, 1)");
    if (!(std::isfinite(p.phase_lobe2_g) && p.phase_lobe2_g > -1.0 &&
          p.phase_lobe2_g < 1.0))
        throw std::runtime_error("phase lobe 2 g must be in (-1, 1)");
    if (!(std::isfinite(p.phase_lobe1_weight) && p.phase_lobe1_weight >= 0.0 &&
          p.phase_lobe1_weight <= 1.0))
        throw std::runtime_error("phase lobe 1 weight must be in [0, 1]");
    if (p.phase_model != "single-hg" && p.phase_model != "mixture-hg")
        throw std::runtime_error("phase model must be single-hg or mixture-hg");
    if (p.phase_model == "single-hg" && p.phase_lobe1_weight != 1.0)
        throw std::runtime_error("single-hg requires phase lobe 1 weight = 1");
    const double phase_first_moment =
        p.phase_lobe1_weight * p.g + (1.0 - p.phase_lobe1_weight) * p.phase_lobe2_g;
    if (std::isfinite(p.expected_phase_first_moment) &&
        std::fabs(p.expected_phase_first_moment - phase_first_moment) > 1.0e-12)
        throw std::runtime_error("phase first moment does not match the supplied lobes");
    if (!(std::isfinite(p.surface_albedo) && p.surface_albedo >= 0.0 &&
          p.surface_albedo <= 1.0))
        throw std::runtime_error("surface albedo must be in [0, 1]");
    if (p.lower_boundary != "lambertian")
        throw std::runtime_error("lower boundary must be lambertian");
    if (p.depth_bins != 0 && p.depth_bins != kStage2DepthBins)
        throw std::runtime_error("depth bins must be 0 or 32");
    if (!(std::isfinite(p.sun_zenith_deg) && p.sun_zenith_deg >= 0.0 &&
          p.sun_zenith_deg < 89.9))
        throw std::runtime_error("sun zenith must be in [0, 89.9)");
    if (!(std::isfinite(p.view_zenith_deg) && p.view_zenith_deg >= 0.0 &&
          p.view_zenith_deg < 89.9))
        throw std::runtime_error("view zenith must be in [0, 89.9)");
    if (!std::isfinite(p.relative_azimuth_deg))
        throw std::runtime_error("relative azimuth must be finite");
    if (p.samples == 0) throw std::runtime_error("samples must be > 0");
    if (p.batch_samples == 0) throw std::runtime_error("batch samples must be > 0");
    if (p.max_scatters <= 0) throw std::runtime_error("max scatters must be > 0");
}

void print_help() {
    std::cout
        << "slab_oracle - deterministic CUDA/CPU plane-parallel cloud oracle\n\n"
        << "Single case:\n"
        << "  slab_oracle [--backend gpu|cpu] [--format json|csv] [--output FILE]\n"
        << "    --tau N --ssa N --g N --sun-zenith-deg N --view-zenith-deg N\n"
        << "    --relative-azimuth-deg N --samples N --seed N --max-scatters N\n"
        << "    --batch-samples N --case NAME\n\n"
        << "Stage-2 reference options (select result schema v2):\n"
        << "    --phase-model single-hg|mixture-hg --phase-lobe1-g N\n"
        << "    --phase-lobe2-g N --phase-lobe1-weight N --phase-first-moment N\n"
        << "    --lower-boundary lambertian --surface-albedo N\n"
        << "    --report-forward-flux true|false --depth-bins 0|32\n\n"
        << "Suites:\n"
        << "  slab_oracle --self-test [--output FILE]\n"
        << "  slab_oracle --sweep [--samples N] [--batch-samples N] [--seed N] [--output FILE]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Parameters parameters;
        std::string backend = "gpu";
        std::string format = "json";
        std::string output_path;
        bool self_test = false;
        bool sweep = false;

        auto value_after = [&](int& index, const char* option) -> std::string {
            if (index + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + option);
            return argv[++index];
        };

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_help();
                return 0;
            } else if (arg == "--self-test") {
                self_test = true;
            } else if (arg == "--sweep") {
                sweep = true;
            } else if (arg == "--backend") {
                backend = value_after(i, "--backend");
            } else if (arg == "--format") {
                format = value_after(i, "--format");
            } else if (arg == "--output") {
                output_path = value_after(i, "--output");
            } else if (arg == "--case") {
                parameters.case_name = value_after(i, "--case");
            } else if (arg == "--tau") {
                parameters.tau = parse_double(value_after(i, "--tau"), "--tau");
            } else if (arg == "--ssa") {
                parameters.ssa = parse_double(value_after(i, "--ssa"), "--ssa");
            } else if (arg == "--g") {
                parameters.g = parse_double(value_after(i, "--g"), "--g");
            } else if (arg == "--phase-model") {
                parameters.phase_model = value_after(i, "--phase-model");
                parameters.stage2_requested = true;
            } else if (arg == "--phase-lobe1-g") {
                parameters.g =
                    parse_double(value_after(i, "--phase-lobe1-g"), "--phase-lobe1-g");
                parameters.stage2_requested = true;
            } else if (arg == "--phase-lobe2-g") {
                parameters.phase_lobe2_g =
                    parse_double(value_after(i, "--phase-lobe2-g"), "--phase-lobe2-g");
                parameters.stage2_requested = true;
            } else if (arg == "--phase-lobe1-weight") {
                parameters.phase_lobe1_weight = parse_double(
                    value_after(i, "--phase-lobe1-weight"), "--phase-lobe1-weight");
                parameters.stage2_requested = true;
            } else if (arg == "--phase-first-moment") {
                parameters.expected_phase_first_moment = parse_double(
                    value_after(i, "--phase-first-moment"), "--phase-first-moment");
                parameters.stage2_requested = true;
            } else if (arg == "--lower-boundary") {
                parameters.lower_boundary = value_after(i, "--lower-boundary");
                parameters.stage2_requested = true;
            } else if (arg == "--surface-albedo") {
                parameters.surface_albedo = parse_double(
                    value_after(i, "--surface-albedo"), "--surface-albedo");
                parameters.stage2_requested = true;
            } else if (arg == "--report-forward-flux") {
                parameters.report_forward_flux = parse_bool(
                    value_after(i, "--report-forward-flux"), "--report-forward-flux");
                parameters.stage2_requested = true;
            } else if (arg == "--depth-bins") {
                parameters.depth_bins =
                    parse_int(value_after(i, "--depth-bins"), "--depth-bins");
                parameters.stage2_requested = true;
            } else if (arg == "--sun-zenith-deg") {
                parameters.sun_zenith_deg =
                    parse_double(value_after(i, "--sun-zenith-deg"), "--sun-zenith-deg");
            } else if (arg == "--view-zenith-deg") {
                parameters.view_zenith_deg =
                    parse_double(value_after(i, "--view-zenith-deg"), "--view-zenith-deg");
            } else if (arg == "--relative-azimuth-deg") {
                parameters.relative_azimuth_deg = parse_double(
                    value_after(i, "--relative-azimuth-deg"), "--relative-azimuth-deg");
            } else if (arg == "--samples") {
                parameters.samples = parse_u64(value_after(i, "--samples"), "--samples");
            } else if (arg == "--seed") {
                parameters.seed = parse_u64(value_after(i, "--seed"), "--seed");
            } else if (arg == "--max-scatters") {
                parameters.max_scatters =
                    parse_int(value_after(i, "--max-scatters"), "--max-scatters");
            } else if (arg == "--batch-samples") {
                parameters.batch_samples =
                    parse_u64(value_after(i, "--batch-samples"), "--batch-samples");
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        }

        if (self_test && sweep) throw std::runtime_error("choose either --self-test or --sweep");
        if (backend != "gpu" && backend != "cpu") throw std::runtime_error("backend must be gpu or cpu");
        if (format != "json" && format != "csv") throw std::runtime_error("format must be json or csv");
        validate(parameters);
        if (sweep && parameters.stage2_requested)
            throw std::runtime_error("the fixed sweep is the legacy v1 contract");
        if (parameters.stage2_requested && format != "json")
            throw std::runtime_error("Stage-2 result schema v2 is JSON-only");

        std::ofstream file;
        std::ostream* output = &std::cout;
        if (!output_path.empty()) {
            file.open(output_path, std::ios::binary | std::ios::trunc);
            if (!file) throw std::runtime_error("could not open output file: " + output_path);
            output = &file;
        }

        if (self_test) {
            return run_self_tests(*output) ? 0 : 2;
        }

        if (sweep) {
            write_csv_header(*output);
            for (const Parameters& p :
                 make_sweep(parameters.samples, parameters.batch_samples, parameters.seed)) {
                validate(p);
                const Result result = backend == "gpu" ? run_gpu(p) : run_cpu(p);
                write_result_csv(*output, result);
                std::cerr << "completed " << p.case_name << ": BRF=" << std::setprecision(8)
                          << result.brf << " +/- " << 1.96 * result.standard_error_brf
                          << ", truncated=" << result.truncated_fraction
                          << ", max_batch_ms=" << result.max_batch_ms << '\n';
            }
            return 0;
        }

        const Result result = backend == "gpu" ? run_gpu(parameters) : run_cpu(parameters);
        if (parameters.stage2_requested) {
            Stage2EnergyResult energy;
            Stage2EnergyResult* energy_ptr = nullptr;
            if (parameters.report_forward_flux || parameters.depth_bins > 0) {
                energy = backend == "gpu" ? run_forward_stage2_gpu(parameters)
                                            : run_forward_stage2_cpu(parameters);
                energy_ptr = &energy;
            }
            write_stage2_result_json(*output, result, energy_ptr);
        } else if (format == "json") {
            write_result_json(*output, result);
        } else {
            write_csv_header(*output);
            write_result_csv(*output, result);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
