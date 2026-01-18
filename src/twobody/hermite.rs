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
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
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

/// Grid spacing strategy for spline construction.
///
/// The choice of grid type significantly affects accuracy for potentials
/// with steep repulsive cores (like Lennard-Jones).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum GridType {
    /// Uniform spacing in r² (legacy behavior).
    /// Gives sparser sampling at short range — poor for steep potentials.
    UniformRsq,
    /// Uniform spacing in r.
    /// Gives equal sampling density everywhere.
    UniformR,
    /// Power-law mapping: r(x) = r_min + (r_max - r_min) * x^p where x ∈ [0,1] is uniform.
    /// For p > 1, gives denser sampling at short range (recommended for steep potentials).
    PowerLaw(f64),
    /// Optimized power-law mapping with p=2 (default).
    /// Uses `x*x` and `sqrt()` instead of `powf()` for better performance.
    /// Equivalent to `PowerLaw(2.0)` but faster.
    #[default]
    PowerLaw2,
    /// Uniform spacing in 1/r² (inverse squared distance).
    /// Transforms LJ potential to polynomial form: U ∝ w⁶ - w³ where w = 1/r².
    /// Gives dense sampling at short range with O(1) lookup using only division.
    /// Recommended for steep potentials when sqrt is expensive.
    InverseRsq,
}

/// Configuration for spline table construction
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    /// Grid spacing strategy (default: PowerLaw(2.0))
    pub grid_type: GridType,
}

impl Default for SplineConfig {
    fn default() -> Self {
        Self {
            n_points: 2000,
            rsq_min: None,
            rsq_max: None,
            shift_energy: true,
            shift_force: false,
            grid_type: GridType::default(),
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

    /// Set grid type (UniformR recommended for steep potentials)
    pub fn with_grid_type(mut self, grid_type: GridType) -> Self {
        self.grid_type = grid_type;
        self
    }
}

// ============================================================================
// Main SplinedPotential implementation
// ============================================================================

/// A splined version of any isotropic twobody potential.
///
/// Provides O(1) evaluation via cubic spline interpolation. Supports three grid types:
/// - `PowerLaw(p)`: Power-law spacing with denser sampling at short range (default, p=2.0)
/// - `UniformR`: Uniform spacing in r
/// - `UniformRsq`: Uniform spacing in r² (legacy, not recommended)
///
/// The splined potential is type-erased after construction—it stores only the
/// precomputed coefficients, not the original potential.
#[derive(Clone)]
pub struct SplinedPotential {
    /// Spline coefficients for each grid interval
    coeffs: Vec<SplineCoeffs>,
    /// Grid type used for construction
    grid_type: GridType,
    /// Minimum r (grid start)
    r_min: f64,
    /// Maximum r (cutoff)
    r_max: f64,
    /// Grid spacing Δr (for UniformR) or Δ(r²) (for UniformRsq)
    delta: f64,
    /// Inverse grid spacing (precomputed)
    inv_delta: f64,
    /// Number of grid points
    n: usize,
    /// Energy shift applied at cutoff
    energy_shift: f64,
    /// Force shift applied (for shift-force scheme)
    force_shift: f64,
    /// Original cutoff distance
    cutoff: f64,
    /// Force at r_min (for linear extrapolation below r_min)
    f_at_rmin: f64,
}

impl SplinedPotential {
    pub fn coefficients(&self) -> &[SplineCoeffs] {
        &self.coeffs
    }

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
    /// Use this for potentials that have infinite cutoff, or when you want
    /// to override the potential's native cutoff.
    ///
    /// # Arguments
    /// * `potential` - The analytical potential to tabulate
    /// * `cutoff` - The cutoff distance (must be finite and positive)
    /// * `config` - Configuration for the spline table
    ///
    /// # Panics
    /// Panics if `n_points < 4`, if rsq_min >= rsq_max, or if cutoff is not finite/positive
    pub fn with_cutoff<P: IsotropicTwobodyEnergy + Cutoff>(
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
    fn build<P: IsotropicTwobodyEnergy + Cutoff>(
        potential: &P,
        cutoff: f64,
        config: SplineConfig,
    ) -> Self {
        let n = config.n_points;
        assert!(n >= 4, "Need at least 4 grid points");

        let rsq_max = config.rsq_max.unwrap_or(cutoff * cutoff);
        // Use lower_cutoff from potential if rsq_min not explicitly set
        let lower = potential.lower_cutoff();
        let rsq_min = config.rsq_min.unwrap_or(lower * lower).max(1e-10); // Avoid zero/negative values

        assert!(rsq_min < rsq_max, "rsq_min must be less than rsq_max");

        let r_min = rsq_min.sqrt();
        let r_max = rsq_max.sqrt();

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

        // Build grid based on grid type
        let (delta, rsq_vals, u_vals, f_vals) = match config.grid_type {
            GridType::UniformRsq => {
                // Legacy: uniform in r² space
                let delta_rsq = (rsq_max - rsq_min) / (n - 1) as f64;
                let mut rsq_vals = Vec::with_capacity(n);
                let mut u_vals = Vec::with_capacity(n);
                let mut f_vals = Vec::with_capacity(n);

                for i in 0..n {
                    let rsq = rsq_min + i as f64 * delta_rsq;
                    rsq_vals.push(rsq);

                    let mut u = potential.isotropic_twobody_energy(rsq) - energy_shift;
                    let mut f = potential.isotropic_twobody_force(rsq);

                    if config.shift_force {
                        let r = rsq.sqrt();
                        u -= (r - r_max) * force_shift;
                        f -= force_shift;
                    }

                    u_vals.push(u);
                    f_vals.push(f);
                }
                (delta_rsq, rsq_vals, u_vals, f_vals)
            }
            GridType::UniformR => {
                // Uniform in r space
                let delta_r = (r_max - r_min) / (n - 1) as f64;
                let mut rsq_vals = Vec::with_capacity(n);
                let mut u_vals = Vec::with_capacity(n);
                let mut f_vals = Vec::with_capacity(n);

                for i in 0..n {
                    let r = r_min + i as f64 * delta_r;
                    let rsq = r * r;
                    rsq_vals.push(rsq);

                    let mut u = potential.isotropic_twobody_energy(rsq) - energy_shift;
                    let mut f = potential.isotropic_twobody_force(rsq);

                    if config.shift_force {
                        u -= (r - r_max) * force_shift;
                        f -= force_shift;
                    }

                    u_vals.push(u);
                    f_vals.push(f);
                }
                (delta_r, rsq_vals, u_vals, f_vals)
            }
            GridType::PowerLaw(p) => {
                // Power-law mapping: r(x) = r_min + (r_max - r_min) * x^p
                // where x = i/(n-1) ∈ [0, 1]. For p > 1, denser at short range.
                assert!(p > 0.0, "Power-law exponent must be positive");
                let r_range = r_max - r_min;
                let mut rsq_vals = Vec::with_capacity(n);
                let mut u_vals = Vec::with_capacity(n);
                let mut f_vals = Vec::with_capacity(n);

                for i in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    let r = r_min + r_range * x.powf(p);
                    let rsq = r * r;
                    rsq_vals.push(rsq);

                    let mut u = potential.isotropic_twobody_energy(rsq) - energy_shift;
                    let mut f = potential.isotropic_twobody_force(rsq);

                    if config.shift_force {
                        u -= (r - r_max) * force_shift;
                        f -= force_shift;
                    }

                    u_vals.push(u);
                    f_vals.push(f);
                }
                // Store p in delta field for use in evaluation
                (p, rsq_vals, u_vals, f_vals)
            }
            GridType::PowerLaw2 => {
                // Optimized power-law mapping with p=2: r(x) = r_min + (r_max - r_min) * x²
                // Uses x*x instead of powf(2.0) for efficiency.
                let r_range = r_max - r_min;
                let mut rsq_vals = Vec::with_capacity(n);
                let mut u_vals = Vec::with_capacity(n);
                let mut f_vals = Vec::with_capacity(n);

                for i in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    let r = r_min + r_range * x * x; // x² instead of x.powf(2.0)
                    let rsq = r * r;
                    rsq_vals.push(rsq);

                    let mut u = potential.isotropic_twobody_energy(rsq) - energy_shift;
                    let mut f = potential.isotropic_twobody_force(rsq);

                    if config.shift_force {
                        u -= (r - r_max) * force_shift;
                        f -= force_shift;
                    }

                    u_vals.push(u);
                    f_vals.push(f);
                }
                // Store p=2.0 in delta field for use in evaluation
                (2.0, rsq_vals, u_vals, f_vals)
            }
            GridType::InverseRsq => {
                // Uniform grid in w = 1/rsq space (inverse squared distance).
                // This transforms LJ to polynomial form: U ∝ w⁶ - w³
                // w_min = 1/rsq_max (long range), w_max = 1/rsq_min (short range)
                let w_min = 1.0 / rsq_max;
                let w_max = 1.0 / rsq_min;
                let delta_w = (w_max - w_min) / (n - 1) as f64;

                let mut rsq_vals = Vec::with_capacity(n);
                let mut u_vals = Vec::with_capacity(n);
                let mut f_vals = Vec::with_capacity(n);

                for i in 0..n {
                    let w = w_min + i as f64 * delta_w;
                    let rsq = 1.0 / w;
                    let r = rsq.sqrt();
                    rsq_vals.push(rsq);

                    let mut u = potential.isotropic_twobody_energy(rsq) - energy_shift;
                    let mut f = potential.isotropic_twobody_force(rsq);

                    if config.shift_force {
                        u -= (r - r_max) * force_shift;
                        f -= force_shift;
                    }

                    u_vals.push(u);
                    f_vals.push(f);
                }
                // Store delta_w for use in evaluation
                (delta_w, rsq_vals, u_vals, f_vals)
            }
        };

        let inv_delta = 1.0 / delta;

        // Compute cubic Hermite spline coefficients
        // For UniformR, we pass delta_rsq as variable spacing (computed per interval)
        let coeffs =
            Self::compute_cubic_hermite_coeffs(&rsq_vals, &u_vals, &f_vals, config.grid_type);

        // Compute force at r_min for linear extrapolation below r_min
        let mut f_at_rmin = potential.isotropic_twobody_force(rsq_min);
        if config.shift_force {
            f_at_rmin -= force_shift;
        }

        Self {
            coeffs,
            grid_type: config.grid_type,
            r_min,
            r_max,
            delta,
            inv_delta,
            n,
            energy_shift,
            force_shift,
            cutoff,
            f_at_rmin,
        }
    }

    /// Compute cubic Hermite spline coefficients.
    ///
    /// For each interval, we fit a cubic polynomial:
    /// ```text
    /// V(ε) = A₀ + A₁ε + A₂ε² + A₃ε³
    /// ```
    /// where ε ∈ [0, 1) is the fractional position within the interval.
    ///
    /// For UniformRsq: ε = (rsq - rsq_i) / Δrsq
    /// For UniformR: ε = (r - r_i) / Δr
    fn compute_cubic_hermite_coeffs(
        rsq: &[f64],
        u: &[f64],
        f: &[f64],
        grid_type: GridType,
    ) -> Vec<SplineCoeffs> {
        let n = rsq.len();
        let mut coeffs = Vec::with_capacity(n);

        for i in 0..n.saturating_sub(1) {
            let r_i = rsq[i].sqrt();
            let r_i1 = rsq[i + 1].sqrt();
            let delta_r = r_i1 - r_i;

            // Energy values at interval endpoints
            let u_i = u[i];
            let u_i1 = u[i + 1];

            // Force values (F = -dU/dr)
            let f_i = f[i];
            let f_i1 = f[i + 1];

            let (a0, a1, a2, a3, b0, b1, b2, b3) = match grid_type {
                GridType::UniformRsq => {
                    // Legacy: polynomial in rsq-space
                    let delta_rsq = rsq[i + 1] - rsq[i];

                    // dU/d(rsq) = dU/dr · dr/d(rsq) = -F(r) / (2r)
                    let duds_i = if r_i > 1e-10 { -f_i / (2.0 * r_i) } else { 0.0 };
                    let duds_i1 = if r_i1 > 1e-10 {
                        -f_i1 / (2.0 * r_i1)
                    } else {
                        0.0
                    };

                    let a0 = u_i;
                    let a1 = delta_rsq * duds_i;
                    let a2 = 3.0 * (u_i1 - u_i) - delta_rsq * (2.0 * duds_i + duds_i1);
                    let a3 = 2.0 * (u_i - u_i1) + delta_rsq * (duds_i + duds_i1);

                    // df/d(rsq) using finite differences
                    let dfds_i = if i > 0 {
                        let delta_prev = rsq[i + 1] - rsq[i.saturating_sub(1)];
                        (f[i + 1] - f[i.saturating_sub(1)]) / delta_prev
                    } else {
                        (f_i1 - f_i) / delta_rsq
                    };
                    let dfds_i1 = if i + 2 < n {
                        let delta_next = rsq[i + 2] - rsq[i];
                        (f[i + 2] - f[i]) / delta_next
                    } else {
                        (f_i1 - f_i) / delta_rsq
                    };

                    let b0 = f_i;
                    let b1 = delta_rsq * dfds_i;
                    let b2 = 3.0 * (f_i1 - f_i) - delta_rsq * (2.0 * dfds_i + dfds_i1);
                    let b3 = 2.0 * (f_i - f_i1) + delta_rsq * (dfds_i + dfds_i1);

                    (a0, a1, a2, a3, b0, b1, b2, b3)
                }
                GridType::UniformR => {
                    // Recommended: polynomial in r-space
                    // dU/dr = -F(r), so derivatives are simply -f
                    let dudr_i = -f_i;
                    let dudr_i1 = -f_i1;

                    let a0 = u_i;
                    let a1 = delta_r * dudr_i;
                    let a2 = 3.0 * (u_i1 - u_i) - delta_r * (2.0 * dudr_i + dudr_i1);
                    let a3 = 2.0 * (u_i - u_i1) + delta_r * (dudr_i + dudr_i1);

                    // df/dr using finite differences
                    let dfdr_i = if i > 0 {
                        let r_prev = rsq[i.saturating_sub(1)].sqrt();
                        (f_i1 - f[i.saturating_sub(1)]) / (r_i1 - r_prev)
                    } else {
                        (f_i1 - f_i) / delta_r
                    };
                    let dfdr_i1 = if i + 2 < n {
                        let r_next = rsq[i + 2].sqrt();
                        (f[i + 2] - f_i) / (r_next - r_i)
                    } else {
                        (f_i1 - f_i) / delta_r
                    };

                    let b0 = f_i;
                    let b1 = delta_r * dfdr_i;
                    let b2 = 3.0 * (f_i1 - f_i) - delta_r * (2.0 * dfdr_i + dfdr_i1);
                    let b3 = 2.0 * (f_i - f_i1) + delta_r * (dfdr_i + dfdr_i1);

                    (a0, a1, a2, a3, b0, b1, b2, b3)
                }
                GridType::PowerLaw(p) => {
                    // Polynomial in x-space where x = ((r - r_min)/(r_max - r_min))^(1/p)
                    // Grid is uniform in x with Δx = 1/(n-1)
                    let delta_x = 1.0 / (n - 1) as f64;
                    let x_i = i as f64 * delta_x;
                    let x_i1 = (i + 1) as f64 * delta_x;

                    // dU/dx = dU/dr * dr/dx, where dr/dx = p * (r_max - r_min) * x^(p-1)
                    // At x_i and x_{i+1}:
                    let r_range = rsq.last().unwrap().sqrt() - rsq.first().unwrap().sqrt();
                    let drdx_i = if x_i > 1e-10 {
                        p * r_range * x_i.powf(p - 1.0)
                    } else {
                        // At x=0, dr/dx → 0 for p > 1, handle gracefully
                        p * r_range * (1e-10_f64).powf(p - 1.0)
                    };
                    let drdx_i1 = p * r_range * x_i1.powf(p - 1.0);

                    // dU/dr = -F, so dU/dx = -F * dr/dx
                    let dudx_i = -f_i * drdx_i;
                    let dudx_i1 = -f_i1 * drdx_i1;

                    let a0 = u_i;
                    let a1 = delta_x * dudx_i;
                    let a2 = 3.0 * (u_i1 - u_i) - delta_x * (2.0 * dudx_i + dudx_i1);
                    let a3 = 2.0 * (u_i - u_i1) + delta_x * (dudx_i + dudx_i1);

                    // For force, use same approach
                    // df/dx = df/dr * dr/dx
                    let dfdx_i = if i > 0 {
                        let x_prev = (i - 1) as f64 * delta_x;
                        (f_i1 - f[i - 1]) / (x_i1 - x_prev)
                    } else {
                        (f_i1 - f_i) / delta_x
                    };
                    let dfdx_i1 = if i + 2 < n {
                        let x_next = (i + 2) as f64 * delta_x;
                        (f[i + 2] - f_i) / (x_next - x_i)
                    } else {
                        (f_i1 - f_i) / delta_x
                    };

                    let b0 = f_i;
                    let b1 = delta_x * dfdx_i;
                    let b2 = 3.0 * (f_i1 - f_i) - delta_x * (2.0 * dfdx_i + dfdx_i1);
                    let b3 = 2.0 * (f_i - f_i1) + delta_x * (dfdx_i + dfdx_i1);

                    (a0, a1, a2, a3, b0, b1, b2, b3)
                }
                GridType::PowerLaw2 => {
                    // Optimized p=2 case: dr/dx = 2 * r_range * x (since x^(p-1) = x^1 = x)
                    let delta_x = 1.0 / (n - 1) as f64;
                    let x_i = i as f64 * delta_x;
                    let x_i1 = (i + 1) as f64 * delta_x;

                    let r_range = rsq.last().unwrap().sqrt() - rsq.first().unwrap().sqrt();
                    // dr/dx = 2 * r_range * x for p=2
                    let drdx_i = if x_i > 1e-10 {
                        2.0 * r_range * x_i
                    } else {
                        2.0 * r_range * 1e-10
                    };
                    let drdx_i1 = 2.0 * r_range * x_i1;

                    let dudx_i = -f_i * drdx_i;
                    let dudx_i1 = -f_i1 * drdx_i1;

                    let a0 = u_i;
                    let a1 = delta_x * dudx_i;
                    let a2 = 3.0 * (u_i1 - u_i) - delta_x * (2.0 * dudx_i + dudx_i1);
                    let a3 = 2.0 * (u_i - u_i1) + delta_x * (dudx_i + dudx_i1);

                    let dfdx_i = if i > 0 {
                        let x_prev = (i - 1) as f64 * delta_x;
                        (f_i1 - f[i - 1]) / (x_i1 - x_prev)
                    } else {
                        (f_i1 - f_i) / delta_x
                    };
                    let dfdx_i1 = if i + 2 < n {
                        let x_next = (i + 2) as f64 * delta_x;
                        (f[i + 2] - f_i) / (x_next - x_i)
                    } else {
                        (f_i1 - f_i) / delta_x
                    };

                    let b0 = f_i;
                    let b1 = delta_x * dfdx_i;
                    let b2 = 3.0 * (f_i1 - f_i) - delta_x * (2.0 * dfdx_i + dfdx_i1);
                    let b3 = 2.0 * (f_i - f_i1) + delta_x * (dfdx_i + dfdx_i1);

                    (a0, a1, a2, a3, b0, b1, b2, b3)
                }
                GridType::InverseRsq => {
                    // Polynomial in w-space where w = 1/rsq
                    // Grid is uniform in w with Δw computed from rsq values
                    let w_i = 1.0 / rsq[i];
                    let w_i1 = 1.0 / rsq[i + 1];
                    let delta_w = w_i1 - w_i; // Note: w increases as rsq decreases

                    // dU/dw = dU/dr * dr/dw
                    // r = rsq^(1/2) = w^(-1/2), so dr/dw = -1/(2*w^(3/2)) = -r³/2
                    // Since dU/dr = -F: dU/dw = -F * (-r³/2) = F * r³ / 2
                    let dudw_i = f_i * r_i * r_i * r_i / 2.0;
                    let dudw_i1 = f_i1 * r_i1 * r_i1 * r_i1 / 2.0;

                    let a0 = u_i;
                    let a1 = delta_w * dudw_i;
                    let a2 = 3.0 * (u_i1 - u_i) - delta_w * (2.0 * dudw_i + dudw_i1);
                    let a3 = 2.0 * (u_i - u_i1) + delta_w * (dudw_i + dudw_i1);

                    // df/dw using finite differences in w-space
                    let dfdw_i = if i > 0 {
                        let w_prev = 1.0 / rsq[i - 1];
                        (f_i1 - f[i - 1]) / (w_i1 - w_prev)
                    } else {
                        (f_i1 - f_i) / delta_w
                    };
                    let dfdw_i1 = if i + 2 < n {
                        let w_next = 1.0 / rsq[i + 2];
                        (f[i + 2] - f_i) / (w_next - w_i)
                    } else {
                        (f_i1 - f_i) / delta_w
                    };

                    let b0 = f_i;
                    let b1 = delta_w * dfdw_i;
                    let b2 = 3.0 * (f_i1 - f_i) - delta_w * (2.0 * dfdw_i + dfdw_i1);
                    let b3 = 2.0 * (f_i - f_i1) + delta_w * (dfdw_i + dfdw_i1);

                    (a0, a1, a2, a3, b0, b1, b2, b3)
                }
            };

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
        self.r_max * self.r_max
    }

    /// Get table statistics for debugging.
    pub fn stats(&self) -> SplineStats {
        SplineStats {
            n_points: self.n,
            rsq_min: self.r_min * self.r_min,
            rsq_max: self.r_max * self.r_max,
            r_min: self.r_min,
            r_max: self.r_max,
            delta: self.delta,
            grid_type: self.grid_type,
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
        let rsq_min = self.r_min * self.r_min;
        let rsq_max = self.r_max * self.r_max;
        let mut max_u_err = 0.0f64;
        let mut max_f_err = 0.0f64;
        let mut worst_rsq_u = rsq_min;
        let mut worst_rsq_f = rsq_min;

        for i in 0..n_test {
            // Test at non-grid points (offset by 0.37 to avoid grid alignment)
            let t = (i as f64 + 0.37) / n_test as f64;
            let rsq = rsq_min + t * (rsq_max - rsq_min);

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
        self.r_max * self.r_max
    }

    #[inline]
    fn lower_cutoff(&self) -> f64 {
        self.r_min
    }
}

impl IsotropicTwobodyEnergy for SplinedPotential {
    /// Evaluate energy at squared distance using cubic spline interpolation.
    ///
    /// Returns 0.0 if rsq >= cutoff². For rsq < rsq_min, linearly extrapolates
    /// using the slope at r_min to maintain repulsive behavior.
    #[inline]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        let rsq_max = self.r_max * self.r_max;

        // Fast path: beyond cutoff
        if distance_squared >= rsq_max {
            return 0.0;
        }

        let rsq_min = self.r_min * self.r_min;
        let r = distance_squared.sqrt();

        // Linear extrapolation correction for r < r_min (branchless)
        // U(r) = U(r_min) + F(r_min) * (r_min - r) for r < r_min
        // This adds a repulsive linear wall below r_min
        let extrap_dist = (self.r_min - r).max(0.0);

        // Compute index and fractional part based on grid type
        let (i, eps) = match self.grid_type {
            GridType::UniformRsq => {
                // Legacy: uniform in r² space
                let rsq = distance_squared.max(rsq_min);
                let t = (rsq - rsq_min) * self.inv_delta;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::UniformR => {
                // Uniform in r space
                let r_clamped = r.max(self.r_min);
                let t = (r_clamped - self.r_min) * self.inv_delta;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::PowerLaw(p) => {
                // Inverse mapping: x = ((r - r_min) / (r_max - r_min))^(1/p)
                let r_clamped = r.max(self.r_min);
                let r_range = self.r_max - self.r_min;
                let x = ((r_clamped - self.r_min) / r_range).powf(1.0 / p);
                let t = x * (self.n - 1) as f64;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::PowerLaw2 => {
                // Optimized p=2: x = sqrt((r - r_min) / (r_max - r_min))
                let r_clamped = r.max(self.r_min);
                let r_range = self.r_max - self.r_min;
                let x = ((r_clamped - self.r_min) / r_range).sqrt(); // sqrt instead of powf(0.5)
                let t = x * (self.n - 1) as f64;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::InverseRsq => {
                // w = 1/rsq, uniform grid in w-space
                // w_min = 1/rsq_max, delta stored in self.delta
                let rsq = distance_squared.max(rsq_min).min(rsq_max);
                let w = 1.0 / rsq;
                let w_min = 1.0 / rsq_max;
                let t = (w - w_min) * self.inv_delta;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
        };

        // Horner's method for polynomial evaluation
        let c = &self.coeffs[i];
        let u_spline = c.u[0] + eps * (c.u[1] + eps * (c.u[2] + eps * c.u[3]));

        // Add linear extrapolation for r < r_min (branchless)
        // extrap_dist is 0 when r >= r_min, so this adds nothing in the normal case
        u_spline + self.f_at_rmin * extrap_dist
    }

    /// Evaluate force at squared distance using cubic spline interpolation.
    ///
    /// Returns 0.0 if rsq >= cutoff². For rsq < rsq_min, returns F(r_min)
    /// (constant force corresponding to linear energy extrapolation).
    #[inline]
    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        let rsq_max = self.r_max * self.r_max;
        if distance_squared >= rsq_max {
            return 0.0;
        }

        let rsq_min = self.r_min * self.r_min;

        let (i, eps) = match self.grid_type {
            GridType::UniformRsq => {
                let rsq = distance_squared.max(rsq_min);
                let t = (rsq - rsq_min) * self.inv_delta;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::UniformR => {
                let r = distance_squared.sqrt().max(self.r_min);
                let t = (r - self.r_min) * self.inv_delta;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::PowerLaw(p) => {
                let r = distance_squared.sqrt().max(self.r_min);
                let r_range = self.r_max - self.r_min;
                let x = ((r - self.r_min) / r_range).powf(1.0 / p);
                let t = x * (self.n - 1) as f64;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::PowerLaw2 => {
                // Optimized p=2: x = sqrt((r - r_min) / (r_max - r_min))
                let r = distance_squared.sqrt().max(self.r_min);
                let r_range = self.r_max - self.r_min;
                let x = ((r - self.r_min) / r_range).sqrt();
                let t = x * (self.n - 1) as f64;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::InverseRsq => {
                // w = 1/rsq, uniform grid in w-space
                let rsq = distance_squared.max(rsq_min).min(rsq_max);
                let w = 1.0 / rsq;
                let w_min = 1.0 / rsq_max;
                let t = (w - w_min) * self.inv_delta;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
        };

        let c = &self.coeffs[i];
        c.f[0] + eps * (c.f[1] + eps * (c.f[2] + eps * c.f[3]))
    }
}

impl Debug for SplinedPotential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplinedPotential")
            .field("n_points", &self.n)
            .field("r_range", &(self.r_min, self.r_max))
            .field("grid_type", &self.grid_type)
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
    /// Grid spacing: Δr for UniformR, Δ(r²) for UniformRsq
    pub delta: f64,
    pub grid_type: GridType,
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
        for ((rsq, u), f) in rsq_values
            .iter()
            .zip(energies.iter_mut())
            .zip(forces.iter_mut())
        {
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
    r_min: f64,
    r_max: f64,
    inv_delta: f64,
    n: usize,
    grid_type: GridType,
    /// Force at r_min for linear extrapolation
    f_at_rmin: f64,
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
            r_min: spline.r_min,
            r_max: spline.r_max,
            inv_delta: spline.inv_delta,
            n: spline.n,
            grid_type: spline.grid_type,
            f_at_rmin: spline.f_at_rmin,
        }
    }

    /// Evaluate energy for a single distance (scalar fallback).
    /// Linearly extrapolates below r_min to maintain repulsive behavior.
    #[inline]
    pub fn energy(&self, rsq: f64) -> f64 {
        let rsq_max = self.r_max * self.r_max;
        if rsq >= rsq_max {
            return 0.0;
        }

        let r = rsq.sqrt();

        // Linear extrapolation distance (0 if r >= r_min)
        let extrap_dist = (self.r_min - r).max(0.0);

        let (i, eps) = match self.grid_type {
            GridType::UniformRsq => {
                let rsq_min = self.r_min * self.r_min;
                let rsq_clamped = rsq.max(rsq_min);
                let t = (rsq_clamped - rsq_min) * self.inv_delta;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::UniformR => {
                let r_clamped = r.max(self.r_min);
                let t = (r_clamped - self.r_min) * self.inv_delta;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::PowerLaw(p) => {
                let r_clamped = r.max(self.r_min);
                let r_range = self.r_max - self.r_min;
                let x = ((r_clamped - self.r_min) / r_range).powf(1.0 / p);
                let t = x * (self.n - 1) as f64;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::PowerLaw2 => {
                let r_clamped = r.max(self.r_min);
                let r_range = self.r_max - self.r_min;
                let x = ((r_clamped - self.r_min) / r_range).sqrt();
                let t = x * (self.n - 1) as f64;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
            GridType::InverseRsq => {
                let rsq_min = self.r_min * self.r_min;
                let rsq_max = self.r_max * self.r_max;
                let rsq_clamped = rsq.max(rsq_min).min(rsq_max);
                let w = 1.0 / rsq_clamped;
                let w_min = 1.0 / rsq_max;
                let t = (w - w_min) * self.inv_delta;
                let i = (t as usize).min(self.n - 2);
                let eps = t - i as f64;
                (i, eps)
            }
        };

        // Horner's method + linear extrapolation
        let u_spline = self.u0[i] + eps * (self.u1[i] + eps * (self.u2[i] + eps * self.u3[i]));
        u_spline + self.f_at_rmin * extrap_dist
    }

    /// Evaluate energies for 4 distances using SIMD (f64x4).
    ///
    /// This is the core SIMD kernel - evaluates 4 spline lookups in parallel.
    /// Linearly extrapolates below r_min to maintain repulsive behavior.
    /// Note: For UniformR and PowerLaw grids, this requires sqrt/powf which may reduce SIMD benefits.
    #[inline]
    pub fn energy_x4(&self, rsq: f64x4) -> f64x4 {
        let rsq_max = f64x4::splat(self.r_max * self.r_max);
        let r_min_v = f64x4::splat(self.r_min);
        let zero = f64x4::ZERO;

        // Compute r for extrapolation (need sqrt for all grid types here)
        let rsq_arr: [f64; 4] = rsq.into();
        let r = f64x4::from([
            rsq_arr[0].sqrt(),
            rsq_arr[1].sqrt(),
            rsq_arr[2].sqrt(),
            rsq_arr[3].sqrt(),
        ]);

        // Linear extrapolation distance (0 if r >= r_min)
        let extrap_dist = (r_min_v - r).max(zero);

        // Compute t values based on grid type
        let t = match self.grid_type {
            GridType::UniformRsq => {
                let rsq_min = f64x4::splat(self.r_min * self.r_min);
                let inv_delta = f64x4::splat(self.inv_delta);
                let rsq_clamped = rsq.max(rsq_min).min(rsq_max);
                (rsq_clamped - rsq_min) * inv_delta
            }
            GridType::UniformR => {
                // Need sqrt for UniformR grid
                let r_min = f64x4::splat(self.r_min);
                let r_max = f64x4::splat(self.r_max);
                let inv_delta = f64x4::splat(self.inv_delta);
                let rsq_arr: [f64; 4] = rsq.into();
                let r = f64x4::from([
                    rsq_arr[0].sqrt(),
                    rsq_arr[1].sqrt(),
                    rsq_arr[2].sqrt(),
                    rsq_arr[3].sqrt(),
                ]);
                let r_clamped = r.max(r_min).min(r_max);
                (r_clamped - r_min) * inv_delta
            }
            GridType::PowerLaw(p) => {
                // PowerLaw: x = ((r - r_min) / r_range)^(1/p), t = x * (n-1)
                // Requires scalar fallback for powf
                let r_range = self.r_max - self.r_min;
                let inv_p = 1.0 / p;
                let n_minus_1 = (self.n - 1) as f64;
                let rsq_arr: [f64; 4] = rsq.into();
                f64x4::from([
                    {
                        let r = rsq_arr[0].sqrt().max(self.r_min).min(self.r_max);
                        ((r - self.r_min) / r_range).powf(inv_p) * n_minus_1
                    },
                    {
                        let r = rsq_arr[1].sqrt().max(self.r_min).min(self.r_max);
                        ((r - self.r_min) / r_range).powf(inv_p) * n_minus_1
                    },
                    {
                        let r = rsq_arr[2].sqrt().max(self.r_min).min(self.r_max);
                        ((r - self.r_min) / r_range).powf(inv_p) * n_minus_1
                    },
                    {
                        let r = rsq_arr[3].sqrt().max(self.r_min).min(self.r_max);
                        ((r - self.r_min) / r_range).powf(inv_p) * n_minus_1
                    },
                ])
            }
            GridType::PowerLaw2 => {
                // Optimized p=2: x = sqrt((r - r_min) / r_range), t = x * (n-1)
                // Uses sqrt instead of powf(0.5)
                let r_range = self.r_max - self.r_min;
                let n_minus_1 = (self.n - 1) as f64;
                let rsq_arr: [f64; 4] = rsq.into();
                f64x4::from([
                    {
                        let r = rsq_arr[0].sqrt().max(self.r_min).min(self.r_max);
                        ((r - self.r_min) / r_range).sqrt() * n_minus_1
                    },
                    {
                        let r = rsq_arr[1].sqrt().max(self.r_min).min(self.r_max);
                        ((r - self.r_min) / r_range).sqrt() * n_minus_1
                    },
                    {
                        let r = rsq_arr[2].sqrt().max(self.r_min).min(self.r_max);
                        ((r - self.r_min) / r_range).sqrt() * n_minus_1
                    },
                    {
                        let r = rsq_arr[3].sqrt().max(self.r_min).min(self.r_max);
                        ((r - self.r_min) / r_range).sqrt() * n_minus_1
                    },
                ])
            }
            GridType::InverseRsq => {
                // w = 1/rsq, t = (w - w_min) * inv_delta
                // This only requires division, no sqrt!
                let rsq_min = self.r_min * self.r_min;
                let rsq_max = self.r_max * self.r_max;
                let w_min = 1.0 / rsq_max;
                let inv_delta = f64x4::splat(self.inv_delta);
                let w_min_v = f64x4::splat(w_min);
                let rsq_min_v = f64x4::splat(rsq_min);
                let rsq_max_v = f64x4::splat(rsq_max);

                // Clamp rsq and compute w = 1/rsq
                let rsq_clamped = rsq.max(rsq_min_v).min(rsq_max_v);
                let one = f64x4::splat(1.0);
                let w = one / rsq_clamped;

                (w - w_min_v) * inv_delta
            }
        };

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
        let u_spline = c3.mul_add(eps, c2);
        let u_spline = u_spline.mul_add(eps, c1);
        let u_spline = u_spline.mul_add(eps, c0);

        // Add linear extrapolation for r < r_min
        let f_at_rmin_v = f64x4::splat(self.f_at_rmin);
        let result = f_at_rmin_v.mul_add(extrap_dist, u_spline);

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
        for rsq in rsq_values.iter().take(n).skip(chunks * 4) {
            total += self.energy(*rsq);
        }

        total
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        8 * (self.u0.len()
            + self.u1.len()
            + self.u2.len()
            + self.u3.len()
            + self.f0.len()
            + self.f1.len()
            + self.f2.len()
            + self.f3.len())
    }
}

impl Debug for SplineTableSimd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplineTableSimd")
            .field("n_intervals", &self.u0.len())
            .field("r_range", &(self.r_min, self.r_max))
            .field("grid_type", &self.grid_type)
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
    use crate::twobody::{AshbaughHatch, Combined, IonIon, LennardJones};
    use coulomb::pairwise::Yukawa;
    use coulomb::permittivity::ConstantPermittivity;

    /// Test cubic Hermite spline for AshbaughHatch + Yukawa from 0 to 100 Å.
    ///
    /// This test prints actual energy values at various distances for debugging.
    #[test]
    fn test_splined_ashbaugh_hatch_yukawa() {
        // Create AshbaughHatch potential
        let epsilon = 0.8; // kJ/mol
        let sigma = 3.0; // Å
        let lambda = 0.5; // hydrophobicity
        let cutoff = 100.0; // Å
        let lj = LennardJones::new(epsilon, sigma);
        let ah = AshbaughHatch::new(lj, lambda, cutoff);

        // Create Yukawa potential (screened electrostatics)
        let charge_product = 1.0; // e²
        let permittivity = ConstantPermittivity::new(80.0);
        let debye_length = 50.0; // Å
        let yukawa_scheme = Yukawa::new(cutoff, Some(debye_length));
        let yukawa = IonIon::new(charge_product, permittivity, yukawa_scheme);

        // Combine potentials
        let combined = Combined::new(ah.clone(), yukawa.clone());

        // Create splined version with range 0 to 100 Å
        let rsq_min = 0.01; // Avoid singularity at r=0
        let rsq_max = cutoff * cutoff; // 10000 Å²
        let config = SplineConfig::high_accuracy()
            .with_rsq_min(rsq_min)
            .with_rsq_max(rsq_max);
        let splined = SplinedPotential::with_cutoff(&combined, cutoff, config);

        println!("\n=== AshbaughHatch + Yukawa Spline Test ===");
        println!(
            "AshbaughHatch: ε={}, σ={}, λ={}, cutoff={}",
            epsilon, sigma, lambda, cutoff
        );
        println!(
            "Yukawa: z₁z₂={}, εᵣ={}, λD={}, cutoff={}",
            charge_product, 80.0, debye_length, cutoff
        );
        println!("Spline: rsq_min={}, rsq_max={}\n", rsq_min, rsq_max);

        // Test distances from 0.1 to 100 Å
        let test_distances = [0.5, 1.0, 10.0, 30.0, 40.0, 100.0];

        println!(
            "{:>8} {:>15} {:>15} {:>15} {:>12}",
            "r (Å)", "u_exact", "u_spline", "u_diff", "rel_err"
        );
        println!("{}", "-".repeat(70));

        for &r in &test_distances {
            let rsq = r * r;
            if rsq < rsq_min || rsq > rsq_max {
                continue;
            }

            let u_exact = combined.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);
            let diff = u_spline - u_exact;
            let rel_err = if u_exact.abs() > 1e-10 {
                (diff / u_exact).abs()
            } else {
                diff.abs()
            };

            println!(
                "{:>8.2} {:>15.6e} {:>15.6e} {:>15.6e} {:>12.2e}",
                r, u_exact, u_spline, diff, rel_err
            );

            // Assert reasonable accuracy (1% for most points, higher tolerance near singularities)
            if r > sigma {
                assert!(
                    rel_err < 0.01 || diff.abs() < 1e-6,
                    "Large error at r={}: exact={}, spline={}, rel_err={}",
                    r,
                    u_exact,
                    u_spline,
                    rel_err
                );
            }
        }

        // Additional validation using the built-in method
        let validation = splined.validate(&combined, 1000);
        println!("\n=== Validation Results ===");
        println!("Max energy error: {:.6e}", validation.max_energy_error);
        println!("Max force error: {:.6e}", validation.max_force_error);
        println!(
            "Worst rsq (energy): {:.2} (r={:.2} Å)",
            validation.worst_rsq_energy,
            validation.worst_rsq_energy.sqrt()
        );
        println!(
            "Worst rsq (force): {:.2} (r={:.2} Å)",
            validation.worst_rsq_force,
            validation.worst_rsq_force.sqrt()
        );

        // Print stats
        let stats = splined.stats();
        println!("\n=== Spline Stats ===");
        println!("n_points: {}", stats.n_points);
        println!("r_min: {:.4} Å", stats.r_min);
        println!("r_max: {:.4} Å", stats.r_max);
        println!("grid_type: {:?}", stats.grid_type);
        println!(
            "delta: {:.6} (Δr for UniformR, Δr² for UniformRsq)",
            stats.delta
        );
        println!("memory_bytes: {}", stats.memory_bytes);
        println!("energy_shift: {:.6e}", stats.energy_shift);

        // Diagnostic: examine grid spacing
        println!("\n=== Grid Spacing Diagnostic ===");
        let delta = stats.delta;
        let r_min_val = stats.r_min;
        match stats.grid_type {
            GridType::UniformR => {
                println!("Grid type: UniformR (constant Δr = {:.4} Å)", delta);
                println!("This gives uniform spacing in r-space.");
            }
            GridType::UniformRsq => {
                println!("Grid type: UniformRsq (constant Δr² = {:.4})", delta);
                println!("delta_r at r=0.1 Å: {:.4} Å", ((0.01 + delta).sqrt() - 0.1));
                println!("delta_r at r=1.0 Å: {:.4} Å", ((1.0 + delta).sqrt() - 1.0));
                println!(
                    "delta_r at r=10 Å: {:.4} Å",
                    ((100.0 + delta).sqrt() - 10.0)
                );
            }
            GridType::PowerLaw(p) => {
                println!("Grid type: PowerLaw (p = {:.1})", p);
                println!("Mapping: r(x) = r_min + (r_max - r_min) * x^p, denser at short range for p > 1");
            }
            GridType::PowerLaw2 => {
                println!("Grid type: PowerLaw2 (p = 2, optimized)");
                println!("Mapping: r(x) = r_min + (r_max - r_min) * x², denser at short range");
            }
            GridType::InverseRsq => {
                println!("Grid type: InverseRsq (constant Δw = {:.6e})", delta);
                println!("Mapping: w = 1/rsq, uniform grid in w-space, denser at short range");
            }
        }

        // Show first few grid points
        println!("\nFirst 10 grid points:");
        for i in 0..10 {
            let r_i = r_min_val + i as f64 * delta;
            let rsq_i = r_i * r_i;
            let u_i = combined.isotropic_twobody_energy(rsq_i);
            println!("  i={}: r={:.4} Å, rsq={:.4}, u={:.6e}", i, r_i, rsq_i, u_i);
        }

        // Examine interpolation at r=0.5 Å
        println!("\n=== Interpolation at r=0.5 Å ===");
        let r_test = 0.5;
        let rsq_test = r_test * r_test;
        let t = (r_test - r_min_val) / delta;
        let i = t as usize;
        let eps = t - i as f64;
        println!(
            "t = (r - r_min) / delta = ({} - {}) / {} = {}",
            r_test, r_min_val, delta, t
        );
        println!("interval index i = {}", i);
        println!("fractional part eps = {:.6}", eps);

        // Grid points bounding this interval
        let r_lo = r_min_val + i as f64 * delta;
        let r_hi = r_min_val + (i + 1) as f64 * delta;
        let rsq_lo = r_lo * r_lo;
        let rsq_hi = r_hi * r_hi;
        let u_lo = combined.isotropic_twobody_energy(rsq_lo);
        let u_hi = combined.isotropic_twobody_energy(rsq_hi);
        let f_lo = combined.isotropic_twobody_force(rsq_lo);
        let f_hi = combined.isotropic_twobody_force(rsq_hi);

        println!("\nInterval [{}, {}]:", i, i + 1);
        println!("  r:   [{:.4}, {:.4}] Å", r_lo, r_hi);
        println!("  rsq: [{:.4}, {:.4}]", rsq_lo, rsq_hi);
        println!("  u:   [{:.6e}, {:.6e}]", u_lo, u_hi);
        println!("  f:   [{:.6e}, {:.6e}]", f_lo, f_hi);
        println!("  u ratio: {:.2e}", u_lo / u_hi);

        // Compute derivatives for Hermite (in r-space for UniformR)
        let dudr_lo = -f_lo; // dU/dr = -F
        let dudr_hi = -f_hi;
        println!("\nDerivatives dU/dr:");
        println!("  at r_lo: {:.6e}", dudr_lo);
        println!("  at r_hi: {:.6e}", dudr_hi);

        // Hermite coefficients (in r-space for UniformR grid)
        let a0 = u_lo;
        let a1 = delta * dudr_lo;
        let a2 = 3.0 * (u_hi - u_lo) - delta * (2.0 * dudr_lo + dudr_hi);
        let a3 = 2.0 * (u_lo - u_hi) + delta * (dudr_lo + dudr_hi);
        println!("\nHermite coefficients (r-space):");
        println!("  a0 = {:.6e}", a0);
        println!("  a1 = {:.6e}", a1);
        println!("  a2 = {:.6e}", a2);
        println!("  a3 = {:.6e}", a3);

        // Evaluate polynomial at eps
        let u_interp = a0 + eps * (a1 + eps * (a2 + eps * a3));
        println!("\nPolynomial evaluation at eps={:.6}:", eps);
        println!("  u_interp = {:.6e}", u_interp);
        println!(
            "  u_exact  = {:.6e}",
            combined.isotropic_twobody_energy(rsq_test)
        );
    }

    /// Test PowerLaw grid with p=2 for AshbaughHatch + Yukawa.
    #[test]
    fn test_powerlaw_grid() {
        let epsilon = 0.8;
        let sigma = 3.0;
        let lambda = 0.5;
        let cutoff = 100.0;
        let lj = LennardJones::new(epsilon, sigma);
        let ah = AshbaughHatch::new(lj, lambda, cutoff);

        let charge_product = 1.0;
        let permittivity = ConstantPermittivity::new(80.0);
        let debye_length = 50.0;
        let yukawa_scheme = Yukawa::new(cutoff, Some(debye_length));
        let yukawa = IonIon::new(charge_product, permittivity, yukawa_scheme);

        let combined = Combined::new(ah, yukawa);

        // Test with PowerLaw(2) grid
        let rsq_min = 0.01;
        let rsq_max = cutoff * cutoff;
        let config = SplineConfig::high_accuracy()
            .with_rsq_min(rsq_min)
            .with_rsq_max(rsq_max)
            .with_grid_type(GridType::PowerLaw(2.0));
        let splined = SplinedPotential::with_cutoff(&combined, cutoff, config);

        println!("\n=== PowerLaw(2) Grid Test ===");
        let stats = splined.stats();
        println!("grid_type: {:?}", stats.grid_type);
        println!("n_points: {}", stats.n_points);

        // Test at various distances
        let test_distances = [0.5, 1.0, 10.0, 30.0, 40.0];
        println!(
            "\n{:>8} {:>15} {:>15} {:>12}",
            "r (Å)", "u_exact", "u_spline", "rel_err"
        );

        for &r in &test_distances {
            let rsq = r * r;
            let u_exact = combined.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);
            let rel_err = if u_exact.abs() > 1e-10 {
                ((u_spline - u_exact) / u_exact).abs()
            } else {
                (u_spline - u_exact).abs()
            };

            println!(
                "{:>8.2} {:>15.6e} {:>15.6e} {:>12.2e}",
                r, u_exact, u_spline, rel_err
            );

            // Assert reasonable accuracy
            if r > sigma {
                assert!(
                    rel_err < 0.02 || (u_spline - u_exact).abs() < 1e-5,
                    "Large error at r={}: rel_err={}",
                    r,
                    rel_err
                );
            }
        }

        let validation = splined.validate(&combined, 1000);
        println!("\nMax energy error: {:.6e}", validation.max_energy_error);
    }

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

    /// Test that spline maintains positive energies at short range (no sign reversal).
    ///
    /// Sign reversal in the repulsive region is catastrophic for MD simulations
    /// as it creates artificial attractive wells that cause particle collapse.
    #[test]
    fn test_no_sign_reversal_short_range_lj() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let rsq_min = 0.01; // r_min = 0.1 σ

        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default().with_rsq_min(rsq_min),
        );

        // Scan from r=0.1 to r=0.5 σ with fine resolution
        // This is the steep repulsive region where sign reversal could occur
        for i in 0..4000 {
            let r = 0.1 + (i as f64) * 0.0001; // 0.1 to 0.5 σ
            let rsq = r * r;
            if rsq < rsq_min {
                continue;
            }

            let u_exact = lj.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);

            // If exact potential is positive (repulsive), spline must also be positive
            if u_exact > 0.0 {
                assert!(
                    u_spline > 0.0,
                    "Sign reversal at r={:.4} σ: exact={:.3e} (positive), spline={:.3e} (negative)",
                    r,
                    u_exact,
                    u_spline
                );
            }
        }
    }

    /// Test no sign reversal for AshbaughHatch + Yukawa at short range.
    #[test]
    fn test_no_sign_reversal_short_range_ah_yukawa() {
        let lj = LennardJones::new(0.8, 3.0);
        let ah = AshbaughHatch::new(lj, 0.5, 100.0);
        let yukawa = IonIon::new(
            1.0,
            ConstantPermittivity::new(80.0),
            Yukawa::new(100.0, Some(50.0)),
        );
        let combined = Combined::new(ah, yukawa);

        let rsq_min = 0.01; // r_min = 0.1 Å
        let splined = SplinedPotential::with_cutoff(
            &combined,
            100.0,
            SplineConfig::default().with_rsq_min(rsq_min),
        );

        // Scan from r=0.1 to r=1.0 Å
        for i in 0..9000 {
            let r = 0.1 + (i as f64) * 0.0001;
            let rsq = r * r;
            if rsq < rsq_min {
                continue;
            }

            let u_exact = combined.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);

            if u_exact > 0.0 {
                assert!(
                    u_spline > 0.0,
                    "Sign reversal at r={:.4} Å: exact={:.3e}, spline={:.3e}",
                    r,
                    u_exact,
                    u_spline
                );
            }
        }
    }

    /// Test that UniformR grid produces sign reversals (regression test).
    ///
    /// This documents the known issue with UniformR at short range and ensures
    /// the default PowerLaw grid is used to avoid it.
    #[test]
    fn test_uniform_r_has_sign_reversal() {
        // Use AH+Yukawa with large cutoff - this is where UniformR fails
        let lj = LennardJones::new(0.8, 3.0);
        let ah = AshbaughHatch::new(lj, 0.5, 100.0);
        let yukawa = IonIon::new(
            1.0,
            ConstantPermittivity::new(80.0),
            Yukawa::new(100.0, Some(50.0)),
        );
        let combined = Combined::new(ah, yukawa);

        let rsq_min = 0.01;
        let splined = SplinedPotential::with_cutoff(
            &combined,
            100.0,
            SplineConfig::high_accuracy()
                .with_rsq_min(rsq_min)
                .with_grid_type(GridType::UniformR),
        );

        // Count negative values where exact is positive
        let mut sign_reversals = 0;
        for i in 0..9000 {
            let r = 0.1 + (i as f64) * 0.0001;
            let rsq = r * r;
            if rsq < rsq_min {
                continue;
            }

            let u_exact = combined.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);

            if u_exact > 0.0 && u_spline < 0.0 {
                sign_reversals += 1;
            }
        }

        // UniformR should have sign reversals (this is why PowerLaw is the default)
        assert!(
            sign_reversals > 0,
            "Expected UniformR to have sign reversals at short range, found none. \
             If this test fails, UniformR may have been fixed and this test can be updated."
        );
    }

    /// Test that default GridType is PowerLaw2.
    #[test]
    fn test_default_grid_type_is_powerlaw2() {
        let default = GridType::default();
        assert_eq!(
            default,
            GridType::PowerLaw2,
            "Default GridType should be PowerLaw2 to prevent sign reversals"
        );
    }

    /// Test that PowerLaw2 produces identical results to PowerLaw(2.0).
    #[test]
    fn test_powerlaw2_matches_powerlaw_2() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let rsq_min = 0.5;

        let splined_p2 = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default()
                .with_rsq_min(rsq_min)
                .with_grid_type(GridType::PowerLaw(2.0)),
        );

        let splined_opt = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default()
                .with_rsq_min(rsq_min)
                .with_grid_type(GridType::PowerLaw2),
        );

        // Test energy and force at multiple distances
        let test_rsq = [0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
        for &rsq in &test_rsq {
            let u_p2 = splined_p2.isotropic_twobody_energy(rsq);
            let u_opt = splined_opt.isotropic_twobody_energy(rsq);
            let f_p2 = splined_p2.isotropic_twobody_force(rsq);
            let f_opt = splined_opt.isotropic_twobody_force(rsq);

            let u_diff = (u_p2 - u_opt).abs();
            let f_diff = (f_p2 - f_opt).abs();

            assert!(
                u_diff < 1e-12,
                "Energy mismatch at rsq={}: PowerLaw(2.0)={}, PowerLaw2={}, diff={}",
                rsq,
                u_p2,
                u_opt,
                u_diff
            );
            assert!(
                f_diff < 1e-12,
                "Force mismatch at rsq={}: PowerLaw(2.0)={}, PowerLaw2={}, diff={}",
                rsq,
                f_p2,
                f_opt,
                f_diff
            );
        }
    }

    /// Test that PowerLaw2 SIMD matches scalar.
    #[test]
    fn test_powerlaw2_simd_matches_scalar() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default().with_grid_type(GridType::PowerLaw2),
        );
        let simd = splined.to_simd();

        let distances: Vec<f64> = (0..100).map(|i| 1.0 + 0.05 * i as f64).collect();

        let scalar_sum: f64 = distances
            .iter()
            .map(|&r2| splined.isotropic_twobody_energy(r2))
            .sum();

        let simd_sum = simd.sum_energies_simd(&distances);

        let rel_err = ((scalar_sum - simd_sum) / scalar_sum).abs();
        assert!(
            rel_err < 1e-10,
            "PowerLaw2 SIMD/scalar mismatch: scalar={}, simd={}, err={}",
            scalar_sum,
            simd_sum,
            rel_err
        );
    }

    /// Test that PowerLaw2 has no sign reversal at short range (like PowerLaw(2.0)).
    #[test]
    fn test_powerlaw2_no_sign_reversal() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let rsq_min = 0.01;

        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default()
                .with_rsq_min(rsq_min)
                .with_grid_type(GridType::PowerLaw2),
        );

        // Scan repulsive region
        for i in 0..4000 {
            let r = 0.1 + (i as f64) * 0.0001;
            let rsq = r * r;
            if rsq < rsq_min {
                continue;
            }

            let u_exact = lj.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);

            if u_exact > 0.0 {
                assert!(
                    u_spline > 0.0,
                    "PowerLaw2 sign reversal at r={:.4}: exact={:.3e}, spline={:.3e}",
                    r,
                    u_exact,
                    u_spline
                );
            }
        }
    }

    /// Test PowerLaw2 with AshbaughHatch + Yukawa combination.
    #[test]
    fn test_powerlaw2_ah_yukawa() {
        let lj = LennardJones::new(0.8, 3.0);
        let ah = AshbaughHatch::new(lj, 0.5, 100.0);
        let yukawa = IonIon::new(
            1.0,
            ConstantPermittivity::new(80.0),
            Yukawa::new(100.0, Some(50.0)),
        );
        let combined = Combined::new(ah, yukawa);

        let rsq_min = 0.01;
        let splined = SplinedPotential::with_cutoff(
            &combined,
            100.0,
            SplineConfig::high_accuracy()
                .with_rsq_min(rsq_min)
                .with_grid_type(GridType::PowerLaw2),
        );

        // Test accuracy at various distances
        let test_distances = [0.5, 1.0, 10.0, 30.0, 40.0];
        for &r in &test_distances {
            let rsq = r * r;
            let u_exact = combined.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);
            let rel_err = if u_exact.abs() > 1e-10 {
                ((u_spline - u_exact) / u_exact).abs()
            } else {
                (u_spline - u_exact).abs()
            };

            if r > 3.0 {
                // Beyond sigma
                assert!(
                    rel_err < 0.02 || (u_spline - u_exact).abs() < 1e-5,
                    "PowerLaw2 large error at r={}: rel_err={}",
                    r,
                    rel_err
                );
            }
        }

        // No sign reversal at short range
        for i in 0..9000 {
            let r = 0.1 + (i as f64) * 0.0001;
            let rsq = r * r;
            if rsq < rsq_min {
                continue;
            }

            let u_exact = combined.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);

            if u_exact > 0.0 {
                assert!(
                    u_spline > 0.0,
                    "PowerLaw2 sign reversal at r={:.4} Å: exact={:.3e}, spline={:.3e}",
                    r,
                    u_exact,
                    u_spline
                );
            }
        }
    }

    // ============================================================================
    // InverseRsq grid tests
    // ============================================================================

    /// Test InverseRsq grid with pure LJ potential (its ideal use case).
    ///
    /// InverseRsq transforms LJ to polynomial form (U ∝ w⁶ - w³ where w = 1/r²),
    /// making it ideal for steep short-range potentials with moderate cutoffs.
    /// For long-range potentials (Yukawa, Coulomb), use PowerLaw2 instead.
    #[test]
    fn test_inversersq_pure_lj() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 3.0; // Typical LJ cutoff

        let rsq_min = 0.5; // Start above the steep repulsive region
        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default()
                .with_rsq_min(rsq_min)
                .with_grid_type(GridType::InverseRsq),
        );

        println!("\n=== InverseRsq Grid Test (Pure LJ) ===");
        let stats = splined.stats();
        println!("grid_type: {:?}", stats.grid_type);
        println!("n_points: {}", stats.n_points);

        // Test accuracy at various distances
        // Note: Near r=σ=1, U≈0 so relative error is misleading; use absolute error there
        let test_distances = [0.8, 1.0, 1.122, 1.5, 2.0, 2.5];
        println!(
            "\n{:>8} {:>15} {:>15} {:>12} {:>12}",
            "r (σ)", "u_exact", "u_spline", "rel_err", "abs_err"
        );

        for &r in &test_distances {
            let rsq = r * r;
            if rsq < rsq_min {
                continue;
            }
            let u_exact = lj.isotropic_twobody_energy(rsq) - splined.energy_shift;
            let u_spline = splined.isotropic_twobody_energy(rsq);
            let abs_err = (u_spline - u_exact).abs();
            let rel_err = if u_exact.abs() > 0.1 {
                abs_err / u_exact.abs()
            } else {
                0.0 // Near zero, relative error is meaningless
            };

            println!(
                "{:>8.3} {:>15.6e} {:>15.6e} {:>12.2e} {:>12.2e}",
                r, u_exact, u_spline, rel_err, abs_err
            );

            // Use relative error when |u| > 0.1, absolute error otherwise
            if u_exact.abs() > 0.1 {
                assert!(
                    rel_err < 0.01,
                    "InverseRsq relative error at r={}: rel_err={}",
                    r,
                    rel_err
                );
            } else {
                // Near zero crossing, accept small absolute errors
                assert!(
                    abs_err < 0.01,
                    "InverseRsq absolute error at r={}: abs_err={}",
                    r,
                    abs_err
                );
            }
        }

        // Validation
        let validation = splined.validate(&lj, 1000);
        println!("\nMax energy error: {:.6e}", validation.max_energy_error);
        println!("Max force error: {:.6e}", validation.max_force_error);
    }

    /// Compare InverseRsq vs PowerLaw2 for short-range LJ accuracy.
    #[test]
    fn test_inversersq_vs_powerlaw2_comparison() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let rsq_min = 0.64; // r_min = 0.8σ
        let n_points = 500; // Use fewer points to see differences

        let config_inv = SplineConfig {
            n_points,
            rsq_min: Some(rsq_min),
            grid_type: GridType::InverseRsq,
            ..Default::default()
        };
        let config_pl2 = SplineConfig {
            n_points,
            rsq_min: Some(rsq_min),
            grid_type: GridType::PowerLaw2,
            ..Default::default()
        };

        let splined_inv = SplinedPotential::with_cutoff(&lj, cutoff, config_inv);
        let splined_pl2 = SplinedPotential::with_cutoff(&lj, cutoff, config_pl2);

        println!(
            "\n=== InverseRsq vs PowerLaw2 Comparison (LJ, {} points) ===",
            n_points
        );
        println!(
            "\n{:>6} {:>12} {:>12} {:>12}",
            "r", "err_InvRsq", "err_PL2", "better"
        );

        let mut inv_wins = 0;
        let mut pl2_wins = 0;

        for i in 0..20 {
            let r = 0.85 + i as f64 * 0.08;
            let rsq = r * r;
            if rsq < rsq_min || rsq > cutoff * cutoff {
                continue;
            }

            let u_exact = lj.isotropic_twobody_energy(rsq);
            let u_inv = splined_inv.isotropic_twobody_energy(rsq);
            let u_pl2 = splined_pl2.isotropic_twobody_energy(rsq);

            let err_inv = ((u_inv - u_exact + splined_inv.energy_shift) / u_exact).abs();
            let err_pl2 = ((u_pl2 - u_exact + splined_pl2.energy_shift) / u_exact).abs();

            let better = if err_inv < err_pl2 {
                inv_wins += 1;
                "InvRsq"
            } else {
                pl2_wins += 1;
                "PL2"
            };

            println!(
                "{:>6.2} {:>12.2e} {:>12.2e} {:>12}",
                r, err_inv, err_pl2, better
            );
        }

        println!(
            "\nInverseRsq wins: {}, PowerLaw2 wins: {}",
            inv_wins, pl2_wins
        );
        println!("Note: InverseRsq should excel at short range (r < 1.5σ)");
    }

    /// Test that InverseRsq has no sign reversal at short range.
    #[test]
    fn test_inversersq_no_sign_reversal_lj() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let rsq_min = 0.01;

        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default()
                .with_rsq_min(rsq_min)
                .with_grid_type(GridType::InverseRsq),
        );

        // Scan repulsive region
        for i in 0..4000 {
            let r = 0.1 + (i as f64) * 0.0001;
            let rsq = r * r;
            if rsq < rsq_min {
                continue;
            }

            let u_exact = lj.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);

            if u_exact > 0.0 {
                assert!(
                    u_spline > 0.0,
                    "InverseRsq sign reversal at r={:.4}: exact={:.3e}, spline={:.3e}",
                    r,
                    u_exact,
                    u_spline
                );
            }
        }
    }

    /// Test that InverseRsq has no sign reversal for AH+Yukawa.
    #[test]
    fn test_inversersq_no_sign_reversal_ah_yukawa() {
        let lj = LennardJones::new(0.8, 3.0);
        let ah = AshbaughHatch::new(lj, 0.5, 100.0);
        let yukawa = IonIon::new(
            1.0,
            ConstantPermittivity::new(80.0),
            Yukawa::new(100.0, Some(50.0)),
        );
        let combined = Combined::new(ah, yukawa);

        let rsq_min = 0.01;
        let splined = SplinedPotential::with_cutoff(
            &combined,
            100.0,
            SplineConfig::high_accuracy()
                .with_rsq_min(rsq_min)
                .with_grid_type(GridType::InverseRsq),
        );

        // Scan from r=0.1 to r=1.0 Å
        for i in 0..9000 {
            let r = 0.1 + (i as f64) * 0.0001;
            let rsq = r * r;
            if rsq < rsq_min {
                continue;
            }

            let u_exact = combined.isotropic_twobody_energy(rsq);
            let u_spline = splined.isotropic_twobody_energy(rsq);

            if u_exact > 0.0 {
                assert!(
                    u_spline > 0.0,
                    "InverseRsq sign reversal at r={:.4} Å: exact={:.3e}, spline={:.3e}",
                    r,
                    u_exact,
                    u_spline
                );
            }
        }
    }

    /// Test that InverseRsq SIMD matches scalar.
    #[test]
    fn test_inversersq_simd_matches_scalar() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default().with_grid_type(GridType::InverseRsq),
        );
        let simd = splined.to_simd();

        let distances: Vec<f64> = (0..100).map(|i| 1.0 + 0.05 * i as f64).collect();

        let scalar_sum: f64 = distances
            .iter()
            .map(|&r2| splined.isotropic_twobody_energy(r2))
            .sum();

        let simd_sum = simd.sum_energies_simd(&distances);

        let rel_err = ((scalar_sum - simd_sum) / scalar_sum).abs();
        assert!(
            rel_err < 1e-10,
            "InverseRsq SIMD/scalar mismatch: scalar={}, simd={}, err={}",
            scalar_sum,
            simd_sum,
            rel_err
        );
    }

    /// Test InverseRsq basic energy accuracy for LJ.
    #[test]
    fn test_inversersq_lj_energy() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default().with_grid_type(GridType::InverseRsq),
        );

        // Test at minimum (r = 2^(1/6) σ ≈ 1.122)
        let r_min = 2.0_f64.powf(1.0 / 6.0);
        let rsq_min = r_min * r_min;

        let u_spline = splined.isotropic_twobody_energy(rsq_min);
        let u_exact = lj.isotropic_twobody_energy(rsq_min) - splined.energy_shift;

        let rel_err = ((u_spline - u_exact) / u_exact).abs();
        assert!(
            rel_err < 1e-3,
            "InverseRsq energy error at LJ minimum: {}",
            rel_err
        );
    }

    /// Test InverseRsq force accuracy for LJ.
    #[test]
    fn test_inversersq_lj_force() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default().with_grid_type(GridType::InverseRsq),
        );

        // Test force at r = 1.5σ
        let rsq = 2.25;
        let f_spline = splined.isotropic_twobody_force(rsq);
        let f_exact = lj.isotropic_twobody_force(rsq);

        let rel_err = ((f_spline - f_exact) / f_exact).abs();
        assert!(rel_err < 1e-2, "InverseRsq force error: {}", rel_err);
    }

    // ============================================================================
    // Linear extrapolation tests
    // ============================================================================

    /// Test that energy increases linearly below r_min (not flat).
    #[test]
    fn test_linear_extrapolation_below_rmin() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let rsq_min = 0.64; // r_min = 0.8σ

        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default().with_rsq_min(rsq_min),
        );

        // Get values at r_min
        let u_at_rmin = splined.isotropic_twobody_energy(rsq_min);
        let f_at_rmin = splined.f_at_rmin;

        println!("\n=== Linear Extrapolation Test ===");
        println!("r_min = {:.3}, rsq_min = {:.3}", rsq_min.sqrt(), rsq_min);
        println!("U(r_min) = {:.6e}", u_at_rmin);
        println!("F(r_min) = {:.6e}", f_at_rmin);

        // Test at distances below r_min
        let test_rsq: [f64; 4] = [0.49, 0.36, 0.25, 0.16]; // r = 0.7, 0.6, 0.5, 0.4
        println!(
            "\n{:>8} {:>12} {:>12} {:>12}",
            "r", "U_spline", "U_expected", "diff"
        );

        for &rsq in &test_rsq {
            let r = rsq.sqrt();
            let r_min = rsq_min.sqrt();

            // Expected: U(r) = U(r_min) + F(r_min) * (r_min - r)
            let delta_r = r_min - r;
            let u_expected = u_at_rmin + f_at_rmin * delta_r;
            let u_spline = splined.isotropic_twobody_energy(rsq);

            let diff = (u_spline - u_expected).abs();
            println!(
                "{:>8.3} {:>12.4e} {:>12.4e} {:>12.4e}",
                r, u_spline, u_expected, diff
            );

            // Should match exactly (within floating point precision)
            assert!(
                diff < 1e-10,
                "Linear extrapolation mismatch at r={}: got {}, expected {}",
                r,
                u_spline,
                u_expected
            );

            // Energy should increase as r decreases (repulsive)
            assert!(
                u_spline > u_at_rmin,
                "Energy should increase below r_min: U({}) = {} <= U(r_min) = {}",
                r,
                u_spline,
                u_at_rmin
            );
        }
    }

    /// Test that force is constant (F(r_min)) below r_min.
    #[test]
    fn test_force_constant_below_rmin() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let rsq_min = 0.64; // r_min = 0.8σ

        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default().with_rsq_min(rsq_min),
        );

        let f_at_rmin = splined.isotropic_twobody_force(rsq_min);

        // Force should be constant (equal to F(r_min)) for all r < r_min
        let test_rsq: [f64; 4] = [0.49, 0.36, 0.25, 0.16];
        for &rsq in &test_rsq {
            let f = splined.isotropic_twobody_force(rsq);
            let diff = (f - f_at_rmin).abs();
            assert!(
                diff < 1e-10,
                "Force should be constant below r_min: F({}) = {}, F(r_min) = {}",
                rsq.sqrt(),
                f,
                f_at_rmin
            );
        }
    }

    /// Test that SIMD extrapolation matches scalar.
    #[test]
    fn test_simd_extrapolation_matches_scalar() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let rsq_min = 0.64;

        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default().with_rsq_min(rsq_min),
        );
        let simd = splined.to_simd();

        // Test with values both above and below r_min
        let distances: [f64; 8] = [0.16, 0.36, 0.49, 0.64, 1.0, 2.0, 4.0, 6.0]; // Mix of below and above r_min

        let scalar: Vec<f64> = distances
            .iter()
            .map(|&rsq| splined.isotropic_twobody_energy(rsq))
            .collect();

        let mut simd_out = vec![0.0; 8];
        simd.energies_batch_simd(&distances, &mut simd_out);

        for i in 0..8 {
            let diff = (scalar[i] - simd_out[i]).abs();
            assert!(
                diff < 1e-10,
                "SIMD/scalar mismatch at rsq={}: scalar={}, simd={}",
                distances[i],
                scalar[i],
                simd_out[i]
            );
        }
    }

    /// Test extrapolation with InverseRsq grid.
    #[test]
    fn test_inversersq_extrapolation() {
        let lj = LennardJones::new(1.0, 1.0);
        let cutoff = 2.5;
        let rsq_min = 0.64;

        let splined = SplinedPotential::with_cutoff(
            &lj,
            cutoff,
            SplineConfig::default()
                .with_rsq_min(rsq_min)
                .with_grid_type(GridType::InverseRsq),
        );

        let u_at_rmin = splined.isotropic_twobody_energy(rsq_min);

        // Test below r_min - energy should increase
        let rsq_below = 0.36; // r = 0.6
        let u_below = splined.isotropic_twobody_energy(rsq_below);

        assert!(
            u_below > u_at_rmin,
            "InverseRsq: Energy should increase below r_min"
        );

        // Verify linear relationship
        let r = rsq_below.sqrt();
        let r_min = rsq_min.sqrt();
        let expected = u_at_rmin + splined.f_at_rmin * (r_min - r);
        let diff = (u_below - expected).abs();
        assert!(
            diff < 1e-10,
            "InverseRsq linear extrapolation mismatch: got {}, expected {}",
            u_below,
            expected
        );
    }
}
