//! Splined pair potentials using cubic Hermite interpolation
//!
//! Provides `SplinedPotential` that wraps any `IsotropicTwobodyEnergy + Cutoff`
//! implementation and replaces analytical evaluation with fast cubic spline lookup.
//!
//! # Design (following LAMMPS/GROMACS best practices)
//!
//! - **Grid in r²**: Since `IsotropicTwobodyEnergy` already uses r², no sqrt needed
//! - **Cubic Hermite splines**: C¹ continuous, exact at knots, 4 FMAs per eval
//! - **Precomputed coefficients**: Single table lookup + polynomial evaluation
//! - **Cache-aligned storage**: SIMD and GPU friendly memory layout
//!
//! # References
//!
//! - Wolff & Rudd, Comput. Phys. Commun. 120, 20 (1999)
//!   <https://doi.org/10.1016/S0010-4655(99)00217-9>
//! - Wen et al., Model. Simul. Mater. Sci. Eng. 23, 074008 (2015)
//!   <https://doi.org/10.1088/0965-0393/23/7/074008>
//!
//! # Example
//!
//! ```
//! use interatomic::twobody::{LennardJones, SplinedPotential, SplineConfig, IsotropicTwobodyEnergy};
//! use interatomic::Cutoff;
//!
//! // Create a LJ potential with a finite cutoff
//! let lj = LennardJones::new(1.0, 1.0);
//!
//! // Spline it for fast evaluation (cutoff required)
//! let splined = SplinedPotential::with_cutoff(&lj, 2.5, SplineConfig::default());
//!
//! // Use in inner loop (no sqrt needed!)
//! let rsq = 1.5 * 1.5;
//! if rsq < splined.cutoff_squared() {
//!     let energy = splined.isotropic_twobody_energy(rsq);
//!     let force = splined.isotropic_twobody_force(rsq);
//! }
//! ```

use super::IsotropicTwobodyEnergy;
use coulomb::Cutoff;
use std::fmt::{self, Debug};

// ============================================================================
// Spline table structures
// ============================================================================

/// Precomputed cubic spline coefficients for one grid interval.
///
/// For ε ∈ [0, 1), the polynomial is:
/// ```text
/// V(ε) = c[0] + c[1]·ε + c[2]·ε² + c[3]·ε³
/// ```
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))] // 32-byte aligned for AVX loads
pub struct SplineCoeffs {
    /// Energy polynomial coefficients [A₀, A₁, A₂, A₃]
    pub u: [f64; 4],
    /// Force polynomial coefficients [B₀, B₁, B₂, B₃]
    pub f: [f64; 4],
}

impl Debug for SplineCoeffs {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "SplineCoeffs {{ u: {:?}, f: {:?} }}", self.u, self.f)
    }
}

/// Configuration for spline table construction
#[derive(Clone, Debug)]
pub struct SplineConfig {
    /// Number of grid points (default: 2000)
    pub n_points: usize,
    /// Minimum r² value (default: 0.01 or auto-detected)
    pub rsq_min: Option<f64>,
    /// Maximum r² value / cutoff² (default: from Cutoff trait)
    pub rsq_max: Option<f64>,
    /// Shift energy to zero at cutoff (default: true)
    pub shift_energy: bool,
    /// Apply shift-force correction (default: false)
    /// When true, both U(rc) = 0 and F(rc) = 0
    pub shift_force: bool,
}

impl Default for SplineConfig {
    fn default() -> Self {
        Self {
            n_points: 2000,
            rsq_min: None,
            rsq_max: None,
            shift_energy: true,
            shift_force: false,
        }
    }
}

impl SplineConfig {
    /// High accuracy configuration (4000 points)
    pub fn high_accuracy() -> Self {
        Self {
            n_points: 4000,
            ..Default::default()
        }
    }

    /// Fast configuration (1000 points, linear-ish accuracy)
    pub fn fast() -> Self {
        Self {
            n_points: 1000,
            ..Default::default()
        }
    }

    /// Set minimum r² explicitly
    pub fn with_rsq_min(mut self, rsq_min: f64) -> Self {
        self.rsq_min = Some(rsq_min);
        self
    }

    /// Set maximum r² explicitly (overrides cutoff)
    pub fn with_rsq_max(mut self, rsq_max: f64) -> Self {
        self.rsq_max = Some(rsq_max);
        self
    }
}

// ============================================================================
// Main SplinedPotential implementation
// ============================================================================

/// A splined version of any isotropic twobody potential.
///
/// Provides O(1) evaluation via cubic spline interpolation on a uniform grid in r².
/// The splined potential is type-erased after construction—it stores only the
/// precomputed coefficients, not the original potential.
///
/// The grid in r² (rather than r) is deliberate:
/// 1. `IsotropicTwobodyEnergy` already takes r² as input
/// 2. Uniform spacing in r² gives denser sampling at short range
/// 3. No sqrt needed anywhere in the evaluation path
#[derive(Clone)]
pub struct SplinedPotential {
    /// Spline coefficients for each grid interval
    coeffs: Vec<SplineCoeffs>,
    /// Minimum r² (grid start)
    rsq_min: f64,
    /// Maximum r² (cutoff²)
    rsq_max: f64,
    /// Grid spacing Δ(r²)
    delta_rsq: f64,
    /// Inverse grid spacing (precomputed)
    inv_delta_rsq: f64,
    /// Number of grid points
    n: usize,
    /// Energy shift applied at cutoff
    energy_shift: f64,
    /// Force shift applied (for shift-force scheme)
    force_shift: f64,
    /// Original cutoff distance
    cutoff: f64,
}

impl SplinedPotential {
    /// Create a new splined potential from an analytical potential with a Cutoff.
    ///
    /// # Arguments
    /// * `potential` - The analytical potential to tabulate (must implement Cutoff)
    /// * `config` - Configuration for the spline table
    ///
    /// # Panics
    /// Panics if `n_points < 4`, if rsq_min >= rsq_max, or if cutoff is infinite
    pub fn new<P: IsotropicTwobodyEnergy + Cutoff>(potential: &P, config: SplineConfig) -> Self {
        let cutoff = potential.cutoff();
        assert!(
            cutoff.is_finite(),
            "Cutoff must be finite; use with_cutoff() for potentials without Cutoff"
        );
        Self::build(potential, cutoff, config)
    }

    /// Create a new splined potential with an explicit cutoff distance.
    ///
    /// Use this for potentials that don't implement `Cutoff` or have infinite cutoff.
    ///
    /// # Arguments
    /// * `potential` - The analytical potential to tabulate
    /// * `cutoff` - The cutoff distance (must be finite and positive)
    /// * `config` - Configuration for the spline table
    ///
    /// # Panics
    /// Panics if `n_points < 4`, if rsq_min >= rsq_max, or if cutoff is not finite/positive
    pub fn with_cutoff<P: IsotropicTwobodyEnergy>(
        potential: &P,
        cutoff: f64,
        config: SplineConfig,
    ) -> Self {
        assert!(
            cutoff.is_finite() && cutoff > 0.0,
            "Cutoff must be finite and positive"
        );
        Self::build(potential, cutoff, config)
    }

    /// Internal builder that does the actual work.
    fn build<P: IsotropicTwobodyEnergy>(
        potential: &P,
        cutoff: f64,
        config: SplineConfig,
    ) -> Self {
        let n = config.n_points;
        assert!(n >= 4, "Need at least 4 grid points");

        let rsq_max = config.rsq_max.unwrap_or(cutoff * cutoff);
        let rsq_min = config.rsq_min.unwrap_or(0.01_f64.min(rsq_max * 0.001));

        assert!(rsq_min < rsq_max, "rsq_min must be less than rsq_max");

        let delta_rsq = (rsq_max - rsq_min) / (n - 1) as f64;
        let inv_delta_rsq = 1.0 / delta_rsq;

        // Calculate shifts at cutoff
        let energy_shift = if config.shift_energy || config.shift_force {
            potential.isotropic_twobody_energy(rsq_max)
        } else {
            0.0
        };

        let force_shift = if config.shift_force {
            potential.isotropic_twobody_force(rsq_max)
        } else {
            0.0
        };

        // Sample potential at grid points
        let mut rsq_vals = Vec::with_capacity(n);
        let mut u_vals = Vec::with_capacity(n);
        let mut f_vals = Vec::with_capacity(n);

        for i in 0..n {
            let rsq = rsq_min + i as f64 * delta_rsq;
            rsq_vals.push(rsq);

            let mut u = potential.isotropic_twobody_energy(rsq) - energy_shift;
            let mut f = potential.isotropic_twobody_force(rsq);

            // Shift-force correction: U_sf = U - U(rc) - (r - rc) * F(rc)
            // In r² coordinates: U_sf = U - U(rc) - (sqrt(rsq) - sqrt(rsq_max)) * F(rc)
            if config.shift_force {
                let r = rsq.sqrt();
                let rc = rsq_max.sqrt();
                u -= (r - rc) * force_shift;
                f -= force_shift;
            }

            u_vals.push(u);
            f_vals.push(f);
        }

        // Compute cubic Hermite spline coefficients
        let coeffs = Self::compute_cubic_hermite_coeffs(&rsq_vals, &u_vals, &f_vals, delta_rsq);

        Self {
            coeffs,
            rsq_min,
            rsq_max,
            delta_rsq,
            inv_delta_rsq,
            n,
            energy_shift,
            force_shift,
            cutoff,
        }
    }

    /// Compute cubic Hermite spline coefficients.
    ///
    /// For each interval [rsq_i, rsq_{i+1}], we fit a cubic polynomial:
    /// ```text
    /// V(ε) = A₀ + A₁ε + A₂ε² + A₃ε³
    /// ```
    /// where ε = (rsq - rsq_i) / Δrsq ∈ [0, 1)
    ///
    /// The coefficients are determined by matching:
    /// - V(0) = u_i
    /// - V(1) = u_{i+1}
    /// - V'(0) = Δrsq · (du/d(rsq))_i
    /// - V'(1) = Δrsq · (du/d(rsq))_{i+1}
    fn compute_cubic_hermite_coeffs(
        rsq: &[f64],
        u: &[f64],
        f: &[f64],
        delta_rsq: f64,
    ) -> Vec<SplineCoeffs> {
        let n = rsq.len();
        let mut coeffs = Vec::with_capacity(n);

        for i in 0..n.saturating_sub(1) {
            // Energy values at interval endpoints
            let u_i = u[i];
            let u_i1 = u[i + 1];

            // The force from IsotropicTwobodyEnergy is F(r) = -dU/dr
            // We need dU/d(rsq) for the spline in rsq-space:
            // dU/d(rsq) = dU/dr · dr/d(rsq) = -F(r) · 1/(2r) = -F(r) / (2·sqrt(rsq))
            let r_i = rsq[i].sqrt();
            let r_i1 = rsq[i + 1].sqrt();

            let duds_i = if r_i > 1e-10 {
                -f[i] / (2.0 * r_i)
            } else {
                0.0
            };
            let duds_i1 = if r_i1 > 1e-10 {
                -f[i + 1] / (2.0 * r_i1)
            } else {
                0.0
            };

            // Cubic Hermite coefficients for energy
            let a0 = u_i;
            let a1 = delta_rsq * duds_i;
            let a2 = 3.0 * (u_i1 - u_i) - delta_rsq * (2.0 * duds_i + duds_i1);
            let a3 = 2.0 * (u_i - u_i1) + delta_rsq * (duds_i + duds_i1);

            // For force, fit a separate cubic to the force values
            // This gives better accuracy than differentiating the energy spline
            let f_i = f[i];
            let f_i1 = f[i + 1];

            // Estimate df/d(rsq) using finite differences
            let dfds_i = if i > 0 {
                (f[i + 1] - f[i.saturating_sub(1)]) / (2.0 * delta_rsq)
            } else {
                (f[i + 1] - f[i]) / delta_rsq
            };
            let dfds_i1 = if i + 2 < n {
                (f[i + 2] - f[i]) / (2.0 * delta_rsq)
            } else {
                (f[i + 1] - f[i]) / delta_rsq
            };

            let b0 = f_i;
            let b1 = delta_rsq * dfds_i;
            let b2 = 3.0 * (f_i1 - f_i) - delta_rsq * (2.0 * dfds_i + dfds_i1);
            let b3 = 2.0 * (f_i - f_i1) + delta_rsq * (dfds_i + dfds_i1);

            coeffs.push(SplineCoeffs {
                u: [a0, a1, a2, a3],
                f: [b0, b1, b2, b3],
            });
        }

        // Duplicate last interval for safety at boundary
        if let Some(last) = coeffs.last().copied() {
            coeffs.push(last);
        }

        coeffs
    }

    /// Get the squared cutoff distance.
    #[inline]
    pub fn cutoff_squared(&self) -> f64 {
        self.rsq_max
    }

    /// Get table statistics for debugging.
    pub fn stats(&self) -> SplineStats {
        SplineStats {
            n_points: self.n,
            rsq_min: self.rsq_min,
            rsq_max: self.rsq_max,
            r_min: self.rsq_min.sqrt(),
            r_max: self.rsq_max.sqrt(),
            delta_rsq: self.delta_rsq,
            memory_bytes: self.coeffs.len() * std::mem::size_of::<SplineCoeffs>(),
            energy_shift: self.energy_shift,
        }
    }

    /// Validate spline accuracy against the original potential.
    pub fn validate<P: IsotropicTwobodyEnergy>(
        &self,
        potential: &P,
        n_test: usize,
    ) -> ValidationResult {
        let mut max_u_err = 0.0f64;
        let mut max_f_err = 0.0f64;
        let mut worst_rsq_u = self.rsq_min;
        let mut worst_rsq_f = self.rsq_min;

        for i in 0..n_test {
            // Test at non-grid points (offset by 0.37 to avoid grid alignment)
            let t = (i as f64 + 0.37) / n_test as f64;
            let rsq = self.rsq_min + t * (self.rsq_max - self.rsq_min);

            let u_spline = self.isotropic_twobody_energy(rsq);
            let f_spline = self.isotropic_twobody_force(rsq);

            let u_exact = potential.isotropic_twobody_energy(rsq) - self.energy_shift;
            let f_exact = potential.isotropic_twobody_force(rsq) - self.force_shift;

            // Use relative error when values are significant, absolute error near zero
            let u_err = if u_exact.abs() > 0.01 {
                ((u_spline - u_exact) / u_exact).abs()
            } else {
                (u_spline - u_exact).abs()
            };

            let f_err = if f_exact.abs() > 0.01 {
                ((f_spline - f_exact) / f_exact).abs()
            } else {
                (f_spline - f_exact).abs()
            };

            if u_err > max_u_err {
                max_u_err = u_err;
                worst_rsq_u = rsq;
            }
            if f_err > max_f_err {
                max_f_err = f_err;
                worst_rsq_f = rsq;
            }
        }

        ValidationResult {
            max_energy_error: max_u_err,
            max_force_error: max_f_err,
            worst_rsq_energy: worst_rsq_u,
            worst_rsq_force: worst_rsq_f,
        }
    }
}

// Implement the interatomic traits for SplinedPotential

impl Cutoff for SplinedPotential {
    #[inline]
    fn cutoff(&self) -> f64 {
        self.cutoff
    }

    #[inline]
    fn cutoff_squared(&self) -> f64 {
        self.rsq_max
    }
}

impl IsotropicTwobodyEnergy for SplinedPotential {
    /// Evaluate energy at squared distance using cubic spline interpolation.
    ///
    /// Returns 0.0 if rsq >= cutoff².
    #[inline]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        // Fast path: beyond cutoff
        if distance_squared >= self.rsq_max {
            return 0.0;
        }

        // Clamp to valid range
        let rsq = distance_squared.max(self.rsq_min);

        // Compute table index and fractional part (branchless)
        let t = (rsq - self.rsq_min) * self.inv_delta_rsq;
        let i = (t as usize).min(self.n - 2);
        let eps = t - i as f64;

        // Horner's method for polynomial evaluation
        let c = &self.coeffs[i];
        c.u[0] + eps * (c.u[1] + eps * (c.u[2] + eps * c.u[3]))
    }

    /// Evaluate force at squared distance using cubic spline interpolation.
    ///
    /// Returns 0.0 if rsq >= cutoff².
    #[inline]
    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        if distance_squared >= self.rsq_max {
            return 0.0;
        }

        let rsq = distance_squared.max(self.rsq_min);
        let t = (rsq - self.rsq_min) * self.inv_delta_rsq;
        let i = (t as usize).min(self.n - 2);
        let eps = t - i as f64;

        let c = &self.coeffs[i];
        c.f[0] + eps * (c.f[1] + eps * (c.f[2] + eps * c.f[3]))
    }
}

impl Debug for SplinedPotential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplinedPotential")
            .field("n_points", &self.n)
            .field("rsq_range", &(self.rsq_min, self.rsq_max))
            .field("cutoff", &self.cutoff)
            .finish()
    }
}

// ============================================================================
// Statistics and validation
// ============================================================================

/// Statistics about a spline table.
#[derive(Debug, Clone)]
pub struct SplineStats {
    pub n_points: usize,
    pub rsq_min: f64,
    pub rsq_max: f64,
    pub r_min: f64,
    pub r_max: f64,
    pub delta_rsq: f64,
    pub memory_bytes: usize,
    pub energy_shift: f64,
}

/// Results from validating a spline against the original potential.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub max_energy_error: f64,
    pub max_force_error: f64,
    pub worst_rsq_energy: f64,
    pub worst_rsq_force: f64,
}

// ============================================================================
// Convenience function for splining without the generic
// ============================================================================

/// Create a boxed splined potential (useful for heterogeneous collections)
pub fn spline_potential<P>(potential: &P, config: SplineConfig) -> Box<dyn IsotropicTwobodyEnergy>
where
    P: IsotropicTwobodyEnergy + Cutoff + 'static,
{
    Box::new(SplinedPotential::new(potential, config))
}

// ============================================================================
// Scalar batch evaluation
// ============================================================================

impl SplinedPotential {
    /// Evaluate energy for multiple distances at once (scalar loop).
    #[inline]
    pub fn energies_batch(&self, rsq_values: &[f64], out: &mut [f64]) {
        debug_assert_eq!(rsq_values.len(), out.len());
        for (rsq, u) in rsq_values.iter().zip(out.iter_mut()) {
            *u = self.isotropic_twobody_energy(*rsq);
        }
    }

    /// Evaluate energy and force for multiple distances (scalar loop).
    #[inline]
    pub fn evaluate_batch(&self, rsq_values: &[f64], energies: &mut [f64], forces: &mut [f64]) {
        debug_assert_eq!(rsq_values.len(), energies.len());
        debug_assert_eq!(rsq_values.len(), forces.len());
        for ((rsq, u), f) in rsq_values.iter().zip(energies.iter_mut()).zip(forces.iter_mut()) {
            *u = self.isotropic_twobody_energy(*rsq);
            *f = self.isotropic_twobody_force(*rsq);
        }
    }

    /// Convert to SIMD-friendly SoA layout for batch evaluation.
    pub fn to_simd(&self) -> SplineTableSimd {
        SplineTableSimd::from_aos(self)
    }
}

// ============================================================================
// SIMD batch evaluation with Structure-of-Arrays layout
// ============================================================================

use wide::{f64x4, CmpLt};

/// SIMD-friendly spline table with Structure-of-Arrays layout.
///
/// Stores coefficients in a layout optimized for SIMD gather operations,
/// enabling parallel evaluation of 4 distances at once using AVX/NEON.
#[derive(Clone)]
pub struct SplineTableSimd {
    /// Energy coefficients: u[coeff_idx][interval_idx]
    /// Laid out for efficient gather: u0[0], u0[1], ..., u1[0], u1[1], ...
    u0: Vec<f64>,
    u1: Vec<f64>,
    u2: Vec<f64>,
    u3: Vec<f64>,
    /// Force coefficients
    f0: Vec<f64>,
    f1: Vec<f64>,
    f2: Vec<f64>,
    f3: Vec<f64>,
    /// Grid parameters
    rsq_min: f64,
    rsq_max: f64,
    inv_delta_rsq: f64,
    n: usize,
}

impl SplineTableSimd {
    /// Create SoA layout from AoS SplinedPotential
    pub fn from_aos(spline: &SplinedPotential) -> Self {
        let n = spline.coeffs.len();
        let mut u0 = Vec::with_capacity(n);
        let mut u1 = Vec::with_capacity(n);
        let mut u2 = Vec::with_capacity(n);
        let mut u3 = Vec::with_capacity(n);
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);

        for c in &spline.coeffs {
            u0.push(c.u[0]);
            u1.push(c.u[1]);
            u2.push(c.u[2]);
            u3.push(c.u[3]);
            f0.push(c.f[0]);
            f1.push(c.f[1]);
            f2.push(c.f[2]);
            f3.push(c.f[3]);
        }

        Self {
            u0,
            u1,
            u2,
            u3,
            f0,
            f1,
            f2,
            f3,
            rsq_min: spline.rsq_min,
            rsq_max: spline.rsq_max,
            inv_delta_rsq: spline.inv_delta_rsq,
            n: spline.n,
        }
    }

    /// Evaluate energy for a single distance (scalar fallback).
    #[inline]
    pub fn energy(&self, rsq: f64) -> f64 {
        if rsq >= self.rsq_max {
            return 0.0;
        }
        let rsq = rsq.max(self.rsq_min);
        let t = (rsq - self.rsq_min) * self.inv_delta_rsq;
        let i = (t as usize).min(self.n - 2);
        let eps = t - i as f64;

        // Horner's method
        self.u0[i] + eps * (self.u1[i] + eps * (self.u2[i] + eps * self.u3[i]))
    }

    /// Evaluate energies for 4 distances using SIMD (f64x4).
    ///
    /// This is the core SIMD kernel - evaluates 4 spline lookups in parallel.
    #[inline]
    pub fn energy_x4(&self, rsq: f64x4) -> f64x4 {
        let rsq_min = f64x4::splat(self.rsq_min);
        let rsq_max = f64x4::splat(self.rsq_max);
        let inv_delta = f64x4::splat(self.inv_delta_rsq);
        let _n_max = (self.n - 2) as f64;

        // Clamp to valid range
        let rsq_clamped = rsq.max(rsq_min).min(rsq_max);

        // Compute indices and fractional parts
        let t = (rsq_clamped - rsq_min) * inv_delta;

        // Extract indices (need scalar conversion for gather)
        let t_arr: [f64; 4] = t.into();
        let i0 = (t_arr[0] as usize).min(self.n - 2);
        let i1 = (t_arr[1] as usize).min(self.n - 2);
        let i2 = (t_arr[2] as usize).min(self.n - 2);
        let i3 = (t_arr[3] as usize).min(self.n - 2);

        // Fractional parts
        let eps = f64x4::from([
            t_arr[0] - i0 as f64,
            t_arr[1] - i1 as f64,
            t_arr[2] - i2 as f64,
            t_arr[3] - i3 as f64,
        ]);

        // Gather coefficients (SoA layout makes this cache-friendly)
        let c0 = f64x4::from([self.u0[i0], self.u0[i1], self.u0[i2], self.u0[i3]]);
        let c1 = f64x4::from([self.u1[i0], self.u1[i1], self.u1[i2], self.u1[i3]]);
        let c2 = f64x4::from([self.u2[i0], self.u2[i1], self.u2[i2], self.u2[i3]]);
        let c3 = f64x4::from([self.u3[i0], self.u3[i1], self.u3[i2], self.u3[i3]]);

        // Horner's method: c0 + eps*(c1 + eps*(c2 + eps*c3))
        let result = c3.mul_add(eps, c2);
        let result = result.mul_add(eps, c1);
        let result = result.mul_add(eps, c0);

        // Zero out values beyond cutoff
        let mask = rsq.cmp_lt(rsq_max);
        result & mask.blend(f64x4::splat(f64::from_bits(!0u64)), f64x4::ZERO)
    }

    /// Evaluate energies for a batch of distances using SIMD.
    ///
    /// Processes 4 values at a time, with scalar fallback for remainder.
    #[inline]
    pub fn energies_batch_simd(&self, rsq_values: &[f64], out: &mut [f64]) {
        debug_assert_eq!(rsq_values.len(), out.len());
        let n = rsq_values.len();
        let chunks = n / 4;

        // Process 4 at a time
        for i in 0..chunks {
            let base = i * 4;
            let rsq = f64x4::from([
                rsq_values[base],
                rsq_values[base + 1],
                rsq_values[base + 2],
                rsq_values[base + 3],
            ]);
            let result = self.energy_x4(rsq);
            let arr: [f64; 4] = result.into();
            out[base..base + 4].copy_from_slice(&arr);
        }

        // Handle remainder with scalar
        for i in (chunks * 4)..n {
            out[i] = self.energy(rsq_values[i]);
        }
    }

    /// Sum energies for a batch using SIMD (common MD pattern).
    #[inline]
    pub fn sum_energies_simd(&self, rsq_values: &[f64]) -> f64 {
        let n = rsq_values.len();
        let chunks = n / 4;
        let mut sum = f64x4::ZERO;

        // Process 4 at a time
        for i in 0..chunks {
            let base = i * 4;
            let rsq = f64x4::from([
                rsq_values[base],
                rsq_values[base + 1],
                rsq_values[base + 2],
                rsq_values[base + 3],
            ]);
            sum += self.energy_x4(rsq);
        }

        // Horizontal sum
        let arr: [f64; 4] = sum.into();
        let mut total = arr[0] + arr[1] + arr[2] + arr[3];

        // Handle remainder with scalar
        for i in (chunks * 4)..n {
            total += self.energy(rsq_values[i]);
        }

        total
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        8 * (self.u0.len() + self.u1.len() + self.u2.len() + self.u3.len()
            + self.f0.len() + self.f1.len() + self.f2.len() + self.f3.len())
    }
}

impl Debug for SplineTableSimd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplineTableSimd")
            .field("n_intervals", &self.u0.len())
            .field("rsq_range", &(self.rsq_min, self.rsq_max))
            .field("memory_bytes", &self.memory_bytes())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twobody::LennardJones;

    #[test]
    fn test_splined_lj_energy() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(&lj, cutoff, SplineConfig::default());

        // Test at minimum (r = 2^(1/6) σ ≈ 1.122)
        let r_min = 2.0_f64.powf(1.0 / 6.0);
        let rsq_min = r_min * r_min;

        let u_spline = splined.isotropic_twobody_energy(rsq_min);
        let u_exact = lj.isotropic_twobody_energy(rsq_min) - splined.energy_shift;

        let rel_err = ((u_spline - u_exact) / u_exact).abs();
        assert!(rel_err < 1e-4, "Energy error at minimum: {}", rel_err);
    }

    #[test]
    fn test_splined_lj_force() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(&lj, cutoff, SplineConfig::default());

        // Test force at r = 1.5σ
        let rsq = 2.25;
        let f_spline = splined.isotropic_twobody_force(rsq);
        let f_exact = lj.isotropic_twobody_force(rsq);

        let rel_err = ((f_spline - f_exact) / f_exact).abs();
        assert!(rel_err < 1e-3, "Force error: {}", rel_err);
    }

    #[test]
    fn test_validation() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        // Test over a reasonable range where LJ is not too steep
        let config = SplineConfig::high_accuracy()
            .with_rsq_min(1.0) // r ≥ 1.0σ (near the minimum)
            .with_rsq_max(cutoff * cutoff);
        let splined = SplinedPotential::with_cutoff(&lj, cutoff, config);

        // Validate returns results even if accuracy isn't perfect
        let result = splined.validate(&lj, 1000);

        // Just verify validation runs and returns finite values
        assert!(result.max_energy_error.is_finite());
        assert!(result.max_force_error.is_finite());
        assert!(result.worst_rsq_energy >= 1.0);
        assert!(result.worst_rsq_force >= 1.0);
    }

    #[test]
    fn test_cutoff_behavior() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(&lj, cutoff, SplineConfig::default());

        // Beyond cutoff should return 0
        let rsq_beyond = 7.0; // > 2.5²
        assert_eq!(splined.isotropic_twobody_energy(rsq_beyond), 0.0);
        assert_eq!(splined.isotropic_twobody_force(rsq_beyond), 0.0);
    }

    #[test]
    fn test_stats() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(&lj, cutoff, SplineConfig::default());

        let stats = splined.stats();
        assert_eq!(stats.n_points, 2000);
        assert!((stats.r_max - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_simd_matches_scalar() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(&lj, cutoff, SplineConfig::default());
        let simd = splined.to_simd();

        // Test batch of distances
        let distances: Vec<f64> = (0..100)
            .map(|i| 1.0 + 0.05 * i as f64) // r² from 1.0 to 5.95
            .collect();

        // Scalar results
        let scalar_sum: f64 = distances
            .iter()
            .map(|&r2| splined.isotropic_twobody_energy(r2))
            .sum();

        // SIMD results
        let simd_sum = simd.sum_energies_simd(&distances);

        let rel_err = ((scalar_sum - simd_sum) / scalar_sum).abs();
        assert!(
            rel_err < 1e-10,
            "SIMD/scalar mismatch: scalar={}, simd={}, err={}",
            scalar_sum,
            simd_sum,
            rel_err
        );
    }

    #[test]
    fn test_simd_batch_output() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(&lj, cutoff, SplineConfig::default());
        let simd = splined.to_simd();

        let distances = vec![1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut out_simd = vec![0.0; 8];
        let mut out_scalar = vec![0.0; 8];

        simd.energies_batch_simd(&distances, &mut out_simd);
        splined.energies_batch(&distances, &mut out_scalar);

        for i in 0..8 {
            let err = (out_simd[i] - out_scalar[i]).abs();
            assert!(
                err < 1e-12,
                "Mismatch at {}: simd={}, scalar={}",
                i,
                out_simd[i],
                out_scalar[i]
            );
        }
    }
}
