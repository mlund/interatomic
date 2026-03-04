// PowerLaw2 spline evaluation for GPU pair potentials
//
// Grid mapping: t = sqrt((r - r_min) / (r_max - r_min)) * (n_coeffs - 1)

struct SplineParams {
    r_min: f32,
    r_max: f32,
    n_coeffs: u32,
    coeff_offset: u32,
    f_at_rmin: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct SplineCoeffs {
    u: vec4<f32>,  // energy coefficients [A0, A1, A2, A3]
    f: vec4<f32>,  // force coefficients  [B0, B1, B2, B3]
}

// Compute spline grid index and fractional epsilon for distance r
fn spline_index_eps(params: SplineParams, r: f32) -> vec2<f32> {
    let r_clamped = clamp(r, params.r_min, params.r_max);
    let r_range = params.r_max - params.r_min;
    let x = sqrt((r_clamped - params.r_min) / r_range);
    let t = x * f32(params.n_coeffs - 1u);
    let i = floor(t);
    return vec2<f32>(min(i, f32(params.n_coeffs - 2u)), t - i);
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

// Compute angle spline grid index and fractional epsilon
fn angle_spline_index_eps(params: AngleSplineParams, angle: f32) -> vec2<f32> {
    let clamped = clamp(angle, params.angle_min, params.angle_max);
    let t = (clamped - params.angle_min) * params.inv_delta;
    let i = floor(t);
    return vec2<f32>(min(i, f32(params.n_coeffs - 2u)), t - i);
}
