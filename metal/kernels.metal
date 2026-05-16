#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------------
// TurboQuant fused kernel (stub — to be implemented)
//
// Pipeline: random rotation → polar encode → pack to 3-bit codes
//           + QJL single-bit residual projection
//
// This stub defines the expected kernel signatures.
// Full implementation requires binding the rotation matrix (R) and
// projection matrix (QJL_R) as Metal buffers.
// ----------------------------------------------------------------------------

struct TurboQuantParams {
    uint dim;
    uint n_bins;      // 2^bits angular bins
    uint batch_size;
};

// Stage 1: rotate input vector and encode into polar codes
kernel void polar_encode(
    device const float16_t* input     [[ buffer(0) ]],
    device const float16_t* R         [[ buffer(1) ]],  // rotation matrix (dim x dim)
    device uint8_t*          codes     [[ buffer(2) ]],
    device float16_t*        norms     [[ buffer(3) ]],
    constant TurboQuantParams& params  [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= params.batch_size) return;

    uint dim = params.dim;
    device const float16_t* x = input + gid * dim;

    // Rotate: x_rot = x @ R  (naive O(dim^2) — use threadgroup for perf)
    float norm_sq = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float val = 0.0f;
        for (uint j = 0; j < dim; j++) {
            val += float(x[j]) * float(R[j * dim + i]);
        }
        // TODO: store x_rot in threadgroup shared memory
        norm_sq += val * val;
    }

    norms[gid] = float16_t(sqrt(norm_sq));
    float inv_norm = 1.0f / (sqrt(norm_sq) + 1e-8f);
    float n_bins = float(params.n_bins);

    for (uint i = 0; i < dim; i++) {
        float val = 0.0f;
        for (uint j = 0; j < dim; j++) {
            val += float(x[j]) * float(R[j * dim + i]);
        }
        float unit = val * inv_norm;
        int code = int((unit + 1.0f) * (n_bins / 2.0f));
        code = clamp(code, 0, int(n_bins) - 1);
        codes[gid * dim + i] = uint8_t(code);
    }
}

// Stage 2: QJL single-bit residual projection
kernel void qjl_project(
    device const float16_t* residual  [[ buffer(0) ]],
    device const float16_t* QJL_R     [[ buffer(1) ]],  // projection matrix (dim x out_dim)
    device bool*             bits      [[ buffer(2) ]],
    constant TurboQuantParams& params  [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= params.batch_size) return;

    uint dim = params.dim;
    device const float16_t* r = residual + gid * dim;

    for (uint i = 0; i < dim; i++) {
        float proj = 0.0f;
        for (uint j = 0; j < dim; j++) {
            proj += float(r[j]) * float(QJL_R[j * dim + i]);
        }
        bits[gid * dim + i] = proj > 0.0f;
    }
}
