// Spline evaluation for GPU pair potentials
//
// Supports two grid types:
//   grid_type == 0u: PowerLaw2  — t = sqrt((r - r_min) / (r_max - r_min)) * (n_coeffs - 1)
//   grid_type == 1u: InverseRsq — t = (1/r² - w_min) * inv_delta

struct SplineParams {
    r_min: f32,
    r_max: f32,
    grid_type: u32,   // 0 = PowerLaw2, 1 = InverseRsq
    n_coeffs: u32,
    coeff_offset: u32,
    inv_delta: f32,
    w_min: f32,       // 1/r_max² (used by InverseRsq)
    f_at_rmin: f32,
}

struct SplineCoeffs {
    u: vec4<f32>,  // energy coefficients [A0, A1, A2, A3]
    f: vec4<f32>,  // force coefficients  [B0, B1, B2, B3]
}

struct AngleSplineParams {
    angle_min: f32,
    angle_max: f32,
    n_coeffs: u32,
    coeff_offset: u32,
    inv_delta: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// Compute spline index and fractional epsilon from distance r
fn spline_index_eps(params: SplineParams, r: f32) -> vec2<f32> {
    var t: f32;
    if (params.grid_type == 0u) {
        // PowerLaw2: t = sqrt((r - r_min) / r_range) * (n - 1)
        let r_clamped = clamp(r, params.r_min, params.r_max);
        let r_range = params.r_max - params.r_min;
        let x = sqrt((r_clamped - params.r_min) / r_range);
        t = x * f32(params.n_coeffs - 1u);
    } else {
        // InverseRsq: t = (1/r² - w_min) * inv_delta
        let rsq = clamp(r * r, params.r_min * params.r_min, params.r_max * params.r_max);
        let w = 1.0 / rsq;
        t = (w - params.w_min) * params.inv_delta;
    }
    let i = floor(t);
    let eps = t - i;
    return vec2<f32>(min(i, f32(params.n_coeffs - 2u)), eps);
}

// Evaluate spline energy using Horner's method: u = A0 + eps*(A1 + eps*(A2 + eps*A3))
fn spline_energy(params: SplineParams, coeffs: array<SplineCoeffs>, r: f32) -> f32 {
    let rsq_max = params.r_max * params.r_max;
    if (r * r >= rsq_max) { return 0.0; }
    let ie = spline_index_eps(params, r);
    let idx = u32(ie.x) + params.coeff_offset;
    let eps = ie.y;
    let c = coeffs[idx].u;
    return c.x + eps * (c.y + eps * (c.z + eps * c.w));
}

// Evaluate spline force using Horner's method: f = B0 + eps*(B1 + eps*(B2 + eps*B3))
fn spline_force(params: SplineParams, coeffs: array<SplineCoeffs>, r: f32) -> f32 {
    let rsq_max = params.r_max * params.r_max;
    if (r * r >= rsq_max) { return 0.0; }
    if (r < params.r_min) { return params.f_at_rmin; }
    let ie = spline_index_eps(params, r);
    let idx = u32(ie.x) + params.coeff_offset;
    let eps = ie.y;
    let c = coeffs[idx].f;
    return c.x + eps * (c.y + eps * (c.z + eps * c.w));
}

// Evaluate both energy and force, returning vec2(energy, force)
fn spline_energy_force(params: SplineParams, coeffs: array<SplineCoeffs>, r: f32) -> vec2<f32> {
    let rsq_max = params.r_max * params.r_max;
    if (r * r >= rsq_max) { return vec2<f32>(0.0, 0.0); }
    let ie = spline_index_eps(params, r);
    let idx = u32(ie.x) + params.coeff_offset;
    let eps = ie.y;
    let cu = coeffs[idx].u;
    let cf = coeffs[idx].f;
    let energy = cu.x + eps * (cu.y + eps * (cu.z + eps * cu.w));
    var force = cf.x + eps * (cf.y + eps * (cf.z + eps * cf.w));
    if (r < params.r_min) { force = params.f_at_rmin; }
    return vec2<f32>(energy, force);
}

// Compute angle spline index and fractional epsilon
fn angle_spline_index_eps(params: AngleSplineParams, angle: f32) -> vec2<f32> {
    let clamped = clamp(angle, params.angle_min, params.angle_max);
    let t = (clamped - params.angle_min) * params.inv_delta;
    let i = floor(t);
    let eps = t - i;
    return vec2<f32>(min(i, f32(params.n_coeffs - 2u)), eps);
}

// Evaluate angular spline energy
fn angle_spline_energy(params: AngleSplineParams, coeffs: array<SplineCoeffs>, angle: f32) -> f32 {
    let ie = angle_spline_index_eps(params, angle);
    let idx = u32(ie.x) + params.coeff_offset;
    let eps = ie.y;
    let c = coeffs[idx].u;
    return c.x + eps * (c.y + eps * (c.z + eps * c.w));
}

// Evaluate angular spline force (torque)
fn angle_spline_force(params: AngleSplineParams, coeffs: array<SplineCoeffs>, angle: f32) -> f32 {
    let ie = angle_spline_index_eps(params, angle);
    let idx = u32(ie.x) + params.coeff_offset;
    let eps = ie.y;
    let c = coeffs[idx].f;
    return c.x + eps * (c.y + eps * (c.z + eps * c.w));
}

// Evaluate both angular energy and force, returning vec2(energy, force)
fn angle_spline_energy_force(params: AngleSplineParams, coeffs: array<SplineCoeffs>, angle: f32) -> vec2<f32> {
    let ie = angle_spline_index_eps(params, angle);
    let idx = u32(ie.x) + params.coeff_offset;
    let eps = ie.y;
    let cu = coeffs[idx].u;
    let cf = coeffs[idx].f;
    let energy = cu.x + eps * (cu.y + eps * (cu.z + eps * cu.w));
    let force = cf.x + eps * (cf.y + eps * (cf.z + eps * cf.w));
    return vec2<f32>(energy, force);
}
