/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#define EPS_AREA 1e-6f
#define EPS_DIST 1e-12f


// backward_exact_forwardmatch.cu
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef PI_F
#define PI_F 3.14159265358979323846f
#endif
#ifndef EPS_DIST
#define EPS_DIST 1e-12f
#endif
#ifndef EPS_AREA
#define EPS_AREA 1e-6f
#endif
#define hashmap_size (1 << 23) 

// -------- float3 helpers --------
__device__ __forceinline__ float3 sub(float3 a, float3 b){ return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ float clamp_m11(float x){ return fmaxf(fminf(x, 1.f), -1.f); }
__device__ __forceinline__ float clamp_gate_minus1_1(float raw){
    return (raw > -1.f && raw < 1.f) ? 1.f : 0.f;
}
__device__ __forceinline__ float sign0(float v){
    return (v > 0.f) ? 1.f : (v < 0.f ? -1.f : 0.f);
}
__device__ inline float3 quat_to_normal(const glm::vec4& q) {
    float x=q.x, y=q.y, z=q.z, w=q.w;
    return { 2.f*(x*z + w*y),
             2.f*(y*z - w*x),
             1.f - 2.f*(x*x + y*y) };
}
__device__ __forceinline__ float4 vjp_quat_from_normal(float4 q, float3 g_n){
    float x=q.x, y=q.y, z=q.z, w=q.w;
    float gx=g_n.x, gy=g_n.y, gz=g_n.z;

    float4 gq;
    gq.x = gx*(2.f*z) + gy*(-2.f*w) + gz*(-4.f*x);
    gq.y = gx*(2.f*w) + gy*( 2.f*z) + gz*(-4.f*y);
    gq.z = gx*(2.f*x) + gy*( 2.f*y) + gz*( 0.f);
    gq.w = gx*(2.f*y) + gy*(-2.f*x) + gz*( 0.f);
    return gq;
}
__device__ __forceinline__ float  dot(float3 a, float3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }


__device__ __forceinline__ float sign_not_zero(float x) {
    return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

__device__ __forceinline__ void atomicAddFloat3(float3* addr, float3 v) {
    atomicAdd(&addr->x, v.x);
    atomicAdd(&addr->y, v.y);
    atomicAdd(&addr->z, v.z);
}
__device__ __forceinline__ void atomicAddVec2(glm::vec2* addr, glm::vec2 v) {
    atomicAdd(&addr->x, v.x);
    atomicAdd(&addr->y, v.y);
}
__device__ inline int compute_hash(int x, int y, int z, int table_size) {
    // 큰 소수를 이용한 공간 해싱
    int h = ((x*73856093) ^ (y*19349663) ^ (z*83492791)) % table_size;
	if(h < 0) h += table_size;
	return h;
}
__device__ __forceinline__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 mul3(float s, float3 a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__device__ __forceinline__ void accum_quat_to_normal_vjp(
    const glm::vec4& q,
    const float3& g_n,
    glm::vec4* g_q
) {
    float x = q.x, y = q.y, z = q.z, w = q.w;

    float gx = g_n.x;
    float gy = g_n.y;
    float gz = g_n.z;

    float dq_x =  2.f * z * gx - 2.f * w * gy - 4.f * x * gz;
    float dq_y =  2.f * w * gx + 2.f * z * gy - 4.f * y * gz;
    float dq_z =  2.f * x * gx + 2.f * y * gy;
    float dq_w =  2.f * y * gx - 2.f * x * gy;

    atomicAdd(&g_q->x, dq_x);
    atomicAdd(&g_q->y, dq_y);
    atomicAdd(&g_q->z, dq_z);
    atomicAdd(&g_q->w, dq_w);
}

__device__ void preprocess_neighbor_effect_backward_exact_forwardmatch(
    int idx,
    float my_envlight,
    float3 my_pos,
    glm::vec2 my_scale,
    float4 my_no,   // xyz = my_normal, w = my_opacity

    const float3* __restrict__ all_points,
    const glm::vec2* __restrict__ all_scales,
    const glm::vec4* __restrict__ all_rotations,
    const float* __restrict__ all_opacities,

    const float* __restrict__ starts,
    const float* __restrict__ ends,
    const float* __restrict__ densities,
    int num_levels,
    int density_threshold,

    const float* __restrict__ neighbor_effects,
    float g,   // dL / d(effect_accum)

    float3* g_my_pos,
    glm::vec2* g_my_scale,
    float4* g_my_no4,

    float* __restrict__ dL_dneighbor_effects,
    glm::vec3* __restrict__ dL_dmean3Ds,
    glm::vec2* __restrict__ dL_dscales,
    glm::vec4* __restrict__ dL_drots,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dgaussian_envlight
){
    const float3 my_normal  = make_float3(my_no.x, my_no.y, my_no.z);
    const float  my_opacity = my_no.w;

    // 현재 forward에서는 사용하지 않는 값들
    (void)my_envlight;
    (void)num_levels;
    (void)density_threshold;
    (void)neighbor_effects;
    (void)dL_dneighbor_effects;
    (void)dL_dgaussian_envlight;

    // forward와 동일
    float cell = 10.f;

    int gx = floorf(my_pos.x / cell);
    int gy = floorf(my_pos.y / cell);
    int gz = floorf(my_pos.z / cell);

    // ============================================================
    // 1) Forward replay
    //
    // forward recurrence:
    //   percent *= percent * (1 - t * abs(val))
    //
    // 즉:
    //   p_new = p_old^2 * (1 - a)
    //   a = t * abs(val)
    // ============================================================
    bool done0 = false;
    float percent = 1.f;

    bool has_last = false;
    int last_ord = -1;
    int last_i = -1;

    for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {

        int ord = ((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1);

        int h = compute_hash(gx + dx, gy + dy, gz + dz, hashmap_size);

        float density = (float)densities[h];
        (void)density;

        int start = (int)starts[h];
        int end   = (int)ends[h];

        for (int i = start; !done0 && i < end; ++i) {
            if (i == idx) continue;

            float3 other_pos      = all_points[i];
            glm::vec2 other_scale = all_scales[i];
            float other_op        = all_opacities[i];
            float3 other_normal   = quat_to_normal(all_rotations[i]);

            float3 o2m = sub(my_pos, other_pos);
            float dist_sq = dot(o2m, o2m);

            if (dist_sq < 8.f) continue;

            float cos_dir = dot(o2m, my_normal);
            float cos_o   = dot(o2m, other_normal);
            (void)cos_o;

            bool dir = (cos_dir <= 0.f);
            if (!dir) continue;

            float cos_area = dot(other_normal, my_normal);

            float area_m = my_opacity * my_scale.x * my_scale.y + EPS_AREA;
            float area_o = other_op * other_scale.x * other_scale.y;

            float raw2 = (area_o / area_m) * cos_area;
            float val  = clamp_m11(raw2);

            float t = 1.f / (1.f + dist_sq);

            percent *= percent * (1.f - t * fabsf(val));

            has_last = true;
            last_ord = ord;
            last_i = i;

            if (percent < 0.0001f) {
                done0 = true;
            }
        }
    }}}

    // forward:
    //   effect_accum += percent;
    //
    // 따라서:
    //   dL / dpercent = g
    float dL_dpercent_new = g;

    float3 l_g_my_pos = make_float3(0.f, 0.f, 0.f);
    glm::vec2 l_g_my_scale(0.f, 0.f);
    float4 l_g_my_no4 = make_float4(0.f, 0.f, 0.f, 0.f);

    // ============================================================
    // 2) Backward replay
    //    forward에서 실제 처리된 prefix만 역순으로 방문
    //
    // forward:
    //   p_new = p_old^2 * (1 - a)
    //
    // reverse reconstruction:
    //   p_old = sqrt(p_new / (1 - a))
    // ============================================================
    if (has_last) {
        float p_new = percent;

        for (int dz = 1; dz >= -1; --dz) {
        for (int dy = 1; dy >= -1; --dy) {
        for (int dx = 1; dx >= -1; --dx) {

            int ord = ((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1);

            if (ord > last_ord) continue;

            int h = compute_hash(gx + dx, gy + dy, gz + dz, hashmap_size);

            int start = (int)starts[h];
            int end   = (int)ends[h];

            for (int i = end - 1; i >= start; --i) {

                if (ord == last_ord && i > last_i) continue;
                if (i == idx) continue;

                float3 other_pos      = all_points[i];
                glm::vec2 other_scale = all_scales[i];
                glm::vec4 other_rot   = all_rotations[i];
                float other_op        = all_opacities[i];
                float3 other_normal   = quat_to_normal(other_rot);

                float3 o2m = sub(my_pos, other_pos);
                float dist_sq = dot(o2m, o2m);

                if (dist_sq < 8.f) continue;

                float cos_dir = dot(o2m, my_normal);
                float cos_o   = dot(o2m, other_normal);
                (void)cos_o;

                bool dir = (cos_dir <= 0.f);
                if (!dir) continue;

                float cos_area = dot(other_normal, my_normal);

                float area_m = my_opacity * my_scale.x * my_scale.y + EPS_AREA;
                float area_o = other_op * other_scale.x * other_scale.y;

                float raw2 = (area_o / area_m) * cos_area;
                float val  = clamp_m11(raw2);

                float t = 1.f / (1.f + dist_sq);
                float abs_val = fabsf(val);

                float a = t * abs_val;
                float one_minus_a = 1.f - a;

                // dist_sq >= 8, t <= 1/9, abs(val) <= 1 이므로
                // one_minus_a는 정상적으로 양수지만 수치 안전장치 유지
                one_minus_a = fmaxf(one_minus_a, 1e-12f);

                float p_old = sqrtf(fmaxf(p_new / one_minus_a, 0.f));

                // ------------------------------------------------
                // p_new = p_old^2 * (1 - a)
                // ------------------------------------------------
                float dL_dp_old = dL_dpercent_new * (2.f * p_old * one_minus_a);
                float dL_da     = dL_dpercent_new * (-(p_old * p_old));

                // a = t * abs(val)
                float dL_dt = dL_da * abs_val;

                float sign_val = 0.f;
                if (val > 0.f) {
                    sign_val = 1.f;
                } else if (val < 0.f) {
                    sign_val = -1.f;
                }

                float dL_dval = dL_da * t * sign_val;

                // ------------------------------------------------
                // val = clamp_m11(raw2)
                // ------------------------------------------------
                float dL_draw2 = 0.f;
                if (raw2 > -1.f && raw2 < 1.f) {
                    dL_draw2 = dL_dval;
                }

                // ------------------------------------------------
                // raw2 = (area_o / area_m) * cos_area
                // ------------------------------------------------
                float inv_area_m = 1.f / area_m;
                float ratio = area_o * inv_area_m;

                float dL_dcos_area = dL_draw2 * ratio;
                float dL_darea_o   = dL_draw2 * (cos_area * inv_area_m);
                float dL_darea_m   = dL_draw2 * (-area_o * cos_area * inv_area_m * inv_area_m);

                // ------------------------------------------------
                // t = 1 / (1 + dist_sq)
                // dt / d_dist_sq = -t^2
                // ------------------------------------------------
                float dL_ddist_sq = dL_dt * (-(t * t));

                // dist_sq = dot(o2m, o2m)
                float3 dL_do2m = mul3(2.f * dL_ddist_sq, o2m);

                // o2m = my_pos - other_pos
                l_g_my_pos = add3(l_g_my_pos, dL_do2m);

                atomicAdd(&dL_dmean3Ds[i].x, -dL_do2m.x);
                atomicAdd(&dL_dmean3Ds[i].y, -dL_do2m.y);
                atomicAdd(&dL_dmean3Ds[i].z, -dL_do2m.z);

                // ------------------------------------------------
                // cos_area = dot(other_normal, my_normal)
                // ------------------------------------------------
                float3 dL_dother_normal = mul3(dL_dcos_area, my_normal);
                float3 dL_dmy_normal    = mul3(dL_dcos_area, other_normal);

                l_g_my_no4.x += dL_dmy_normal.x;
                l_g_my_no4.y += dL_dmy_normal.y;
                l_g_my_no4.z += dL_dmy_normal.z;

                accum_quat_to_normal_vjp(other_rot, dL_dother_normal, &dL_drots[i]);

                // ------------------------------------------------
                // area_o = other_op * other_scale.x * other_scale.y
                // ------------------------------------------------
                float dL_dother_op = dL_darea_o * (other_scale.x * other_scale.y);
                float dL_dother_sx = dL_darea_o * (other_op * other_scale.y);
                float dL_dother_sy = dL_darea_o * (other_op * other_scale.x);

                atomicAdd(&dL_dopacity[i], dL_dother_op);
                atomicAdd(&dL_dscales[i].x, dL_dother_sx);
                atomicAdd(&dL_dscales[i].y, dL_dother_sy);

                // ------------------------------------------------
                // area_m = my_opacity * my_scale.x * my_scale.y + EPS_AREA
                // ------------------------------------------------
                l_g_my_no4.w   += dL_darea_m * (my_scale.x * my_scale.y);
                l_g_my_scale.x += dL_darea_m * (my_opacity * my_scale.y);
                l_g_my_scale.y += dL_darea_m * (my_opacity * my_scale.x);

                // reverse recurrence
                p_new = p_old;
                dL_dpercent_new = dL_dp_old;
            }
        }}}
    }

    g_my_pos->x += l_g_my_pos.x;
    g_my_pos->y += l_g_my_pos.y;
    g_my_pos->z += l_g_my_pos.z;

    g_my_scale->x += l_g_my_scale.x;
    g_my_scale->y += l_g_my_scale.y;

    g_my_no4->x += l_g_my_no4.x;
    g_my_no4->y += l_g_my_no4.y;
    g_my_no4->z += l_g_my_no4.z;
    g_my_no4->w += l_g_my_no4.w;
}
// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* avg_neighbor,
	const float* neighbor_shade,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dneighbor_shade)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float collected_neighbor_effect[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };//backward에서 쓰는 “이전까지의 누적색(재구성)” 버퍼 
	float dL_dpixel[C];

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;
	float dL_dneighbor_effect= 0.f;
	float avg_effect=0.f;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		dL_dneighbor_effect = dL_depths[NEIGHBOR_EFFECT * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];
		avg_effect=avg_neighbor[pix_id];
		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	const float neighbor_denom = fmaxf(final_A, 1e-6f);
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	
	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			collected_neighbor_effect[block.thread_rank()] = neighbor_shade[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussiansx
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			const float effect = collected_neighbor_effect[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (c_d < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// accumulations

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			float G = exp(power);
			float alpha = min(0.99f, opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			float w = alpha * T;


			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];

				// forward color: C = sum_i color_i * alpha_i * T_i
				// reverse compositing reconstruction

				const float dL_dchannel = dL_dpixel[ch];
				
				if (ch == 3)
				{
					// forward:
					// y3 = sum_i c_i * w_i / S
					atomicAdd(
						&(dL_dcolors[global_id * C + ch]),
						dL_dchannel * w / neighbor_denom
					);

					dL_dweight +=
						dL_dchannel * (c - avg_color3) / neighbor_denom;

					continue;
				}
				if (ch > 3) {continue;}
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;
				// d pixel / d alpha_i
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// d pixel / d color_i = alpha_i * T_i = w
				atomicAdd(&(dL_dcolors[global_id * C + ch]), w * dL_dchannel);
			}
			float dL_dz = 0.0f;
			float dL_dweight = 0;
#if RENDER_AXUTILITY
			if (neighbor_denom > 1e-6f)
			{
				// d avg / d effect_i = w_i / sum(w)
				atomicAdd(
					&(dL_dneighbor_shade[global_id]),
					dL_dneighbor_effect * w / neighbor_denom
				);

				// d avg / d w_i = (effect_i - avg_effect) / sum(w)
				dL_dweight += dL_dneighbor_effect * (effect - avg_effect) / neighbor_denom;
			}
			

			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT 
			// if not detached weight, sometimes 
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend


			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth; 
#endif

			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};
				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz); // propagate depth loss
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}


__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float3* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec2* dL_dscales,
	 glm::vec4* dL_drots)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
			glm::vec4(L[0], 0.0),
			glm::vec4(L[1], 0.0),
			glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
		);

		glm::mat4 world2ndc = glm::mat4(
			projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
			projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
			projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
			projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
		);

		glm::mat3x4 ndc2pix = glm::mat3x4(
			glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
			glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		);

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float* opacities,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 

	const float* gaussian_envlight,
	const float*  starts,
    const float*  ends,
    const float*   densities,
	const float*  neighbor_effects,
	const float*  neighbor_shade,
	// grad input
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_dshs,
	float3* dL_dmean2Ds,
	float* dL_dneighbor_shade,

	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	float*  dL_dopacity,
    float* dL_dneighbor_effects,
	float* dL_dgaussian_envlight
)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots
	);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
	
	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc

	float g = dL_dneighbor_shade[idx];
    if(g == 0.f) return;

	float3 my_pos   = means3D[idx];
	glm::vec2 my_scale = scales[idx];
	float3 my_normal = quat_to_normal(rotations[idx]);
	float  my_opacity = opacities[idx];
	float my_envlight= gaussian_envlight[idx];

	int num_levels=1;
	int	density_threshold=20;

	float4 my_no = {my_normal.x, my_normal.y, my_normal.z, my_opacity};

	// local gradient accumulators
	float3   g_my_pos   = make_float3(0,0,0);
	glm::vec2 g_my_scale(0.f, 0.f);
	float4   g_my_no4   = make_float4(0,0,0,0);

	preprocess_neighbor_effect_backward_exact_forwardmatch(
		idx,
	    my_envlight,

		my_pos,
		my_scale,
		my_no,

		means3D,      // all_points
		scales,       // const glm::vec2*
		rotations,    // const glm::vec4*
		opacities,

		starts,
		ends,
		densities,
		num_levels,
		density_threshold,

		neighbor_effects,
		g,

		&g_my_pos,
		&g_my_scale,
		&g_my_no4,

		dL_dneighbor_effects,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots,
		dL_dopacity,
		dL_dgaussian_envlight
);
	dL_dmean3Ds[idx].x += g_my_pos.x;
	dL_dmean3Ds[idx].y += g_my_pos.y;
	dL_dmean3Ds[idx].z += g_my_pos.z;
	dL_dscales[idx].x += g_my_scale.x;
	dL_dscales[idx].y += g_my_scale.y;
    dL_dopacity[idx] += g_my_no4.w;

// rotation grad: g_my_no4.xyz (dL/dnormal) -> dL/dquat
	float4 q_my = make_float4(rotations[idx].x, rotations[idx].y, rotations[idx].z, rotations[idx].w);
	float3 g_my_n = make_float3(g_my_no4.x, g_my_no4.y, g_my_no4.z);
	float4 gq_my = vjp_quat_from_normal(q_my, g_my_n);

	dL_drots[idx].x += gq_my.x;
	dL_drots[idx].y += gq_my.y;
	dL_drots[idx].z += gq_my.z;
	dL_drots[idx].w += gq_my.w;
}

void BACKWARD:: preprocess(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float* opacities,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 

	const float* gaussian_envlight,
	const float*  starts,
    const float*  ends,
    const float*   densities,
	const float*  neighbor_effects,
	const float* neighbor_shade, 
	// grad inpu
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_dshs,
	float3* dL_dmean2Ds,
	float* dL_dneighbor_shade,

	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots,
	float*  dL_dopacity,
    float* dL_dneighbor_effects,
	float*  dL_dgaussian_envlight
)
{	
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		opacities,
	    scale_modifier,
	    viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		(glm::vec3*) campos, 

		gaussian_envlight,
		starts,
		ends,
		densities,
		neighbor_effects,
		neighbor_shade, 
		// grad input
		dL_dnormal3Ds,
		dL_dtransMats,
		dL_dcolors,
		dL_dshs,
		(float3*) dL_dmean2Ds,
		dL_dneighbor_shade,

		(glm::vec3*) dL_dmean3Ds,
		(glm::vec2*) dL_dscales,
		(glm::vec4*) dL_drots,
		dL_dopacity,
		dL_dneighbor_effects,
		dL_dgaussian_envlight
	);
}
	
void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* bg_color,
	const float2* means2D,
	const float4* normal_opacity,
	const float* transMats,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* avg_neighbor,
	const float* neighbor_shade,
	const float* dL_dpixels,
	const float* dL_depths,
	float * dL_dtransMat,
	float3* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dneighbor_shade)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		bg_color,
		means2D,
		normal_opacity,
		transMats,
		colors,
		depths,
		final_Ts,
		n_contrib,
		avg_neighbor,
		neighbor_shade,
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors,
		dL_dneighbor_shade
		);
}
