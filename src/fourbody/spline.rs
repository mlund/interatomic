//! Splined dihedral potential using cubic Hermite interpolation.
//!
//! Wraps any `FourbodyAngleEnergy` implementation with a fast cubic spline
//! lookup over the dihedral angle range [-π, π]. Uses a uniform grid in angle space.

use super::FourbodyAngleEnergy;
use crate::twobody::{compute_uniform_hermite_coeffs, SplineCoeffs};
use std::f64::consts::PI;

/// A splined version of any fourbody dihedral potential.
///
/// Provides O(1) evaluation via cubic Hermite interpolation over [-π, π].
/// Implements `FourbodyAngleEnergy` as a drop-in replacement.
#[derive(Clone, Debug)]
pub struct SplinedDihedral {
    coeffs: Vec<SplineCoeffs>,
    inv_delta: f64,
}

impl SplinedDihedral {
    /// Minimum angle of the grid (always -π).
    const ANGLE_MIN: f64 = -PI;
    /// Maximum angle of the grid (always π).
    const ANGLE_MAX: f64 = PI;

    /// Create a splined dihedral potential from an analytical potential.
    ///
    /// # Arguments
    /// * `potential` - The analytical dihedral potential to tabulate
    /// * `n_points` - Number of grid points (must be >= 4)
    ///
    /// # Panics
    /// Panics if `n_points < 4`.
    pub fn new(potential: &dyn FourbodyAngleEnergy, n_points: usize) -> Self {
        assert!(n_points >= 4, "Need at least 4 grid points");

        let delta = (Self::ANGLE_MAX - Self::ANGLE_MIN) / (n_points - 1) as f64;

        // Evaluate potential at grid points
        let mut u_vals = Vec::with_capacity(n_points);
        let mut f_vals = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let angle = Self::ANGLE_MIN + i as f64 * delta;
            u_vals.push(potential.fourbody_angle_energy(angle));
            f_vals.push(potential.fourbody_angle_force(angle));
        }

        let coeffs = compute_uniform_hermite_coeffs(&u_vals, &f_vals, delta);

        Self {
            coeffs,
            inv_delta: 1.0 / delta,
        }
    }

    /// Returns a reference to the spline coefficients.
    pub fn coefficients(&self) -> &[SplineCoeffs] {
        &self.coeffs
    }

    /// Returns the minimum angle of the grid.
    pub const fn angle_min(&self) -> f64 {
        Self::ANGLE_MIN
    }

    /// Returns the maximum angle of the grid.
    pub const fn angle_max(&self) -> f64 {
        Self::ANGLE_MAX
    }

    /// Returns the inverse grid spacing.
    pub const fn inv_delta(&self) -> f64 {
        self.inv_delta
    }

    /// Evaluate both energy and force in a single lookup.
    #[inline(always)]
    pub fn energy_and_force(&self, angle: f64) -> (f64, f64) {
        let (i, eps) = self.index_eps(angle);
        let c = &self.coeffs[i];
        let energy = c.u[0] + eps * (c.u[1] + eps * (c.u[2] + eps * c.u[3]));
        let force = c.f[0] + eps * (c.f[1] + eps * c.f[2]);
        (energy, force)
    }

    /// Compute index and epsilon for a given angle.
    #[inline(always)]
    fn index_eps(&self, angle: f64) -> (usize, f64) {
        let clamped = angle.clamp(Self::ANGLE_MIN, Self::ANGLE_MAX);
        let t = (clamped - Self::ANGLE_MIN) * self.inv_delta;
        let i = (t as usize).min(self.coeffs.len() - 2);
        let eps = t - i as f64;
        (i, eps)
    }
}

impl FourbodyAngleEnergy for SplinedDihedral {
    #[inline(always)]
    fn fourbody_angle_energy(&self, angle: f64) -> f64 {
        let (i, eps) = self.index_eps(angle);
        let c = &self.coeffs[i];
        c.u[0] + eps * (c.u[1] + eps * (c.u[2] + eps * c.u[3]))
    }

    #[inline(always)]
    fn fourbody_angle_force(&self, angle: f64) -> f64 {
        let (i, eps) = self.index_eps(angle);
        let c = &self.coeffs[i];
        c.f[0] + eps * (c.f[1] + eps * c.f[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourbody::{HarmonicDihedral, PeriodicDihedral};
    use approx::assert_relative_eq;

    #[test]
    fn test_splined_dihedral_harmonic_energy() {
        let pot = HarmonicDihedral::new(0.0, 50.0);
        let splined = SplinedDihedral::new(&pot, 500);

        let test_angles = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        for &angle in &test_angles {
            let exact = pot.fourbody_angle_energy(angle);
            let approx = splined.fourbody_angle_energy(angle);
            let rel_err = if exact.abs() > 1e-10 {
                ((approx - exact) / exact).abs()
            } else {
                (approx - exact).abs()
            };
            assert!(
                rel_err < 1e-6,
                "Energy error too large at angle={angle}: exact={exact}, approx={approx}, rel_err={rel_err}",
            );
        }
    }

    #[test]
    fn test_splined_dihedral_harmonic_force() {
        let pot = HarmonicDihedral::new(0.5, 100.0);
        let splined = SplinedDihedral::new(&pot, 500);

        let test_angles = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 2.5];
        for &angle in &test_angles {
            let exact = pot.fourbody_angle_force(angle);
            let approx = splined.fourbody_angle_force(angle);
            let rel_err = if exact.abs() > 1e-10 {
                ((approx - exact) / exact).abs()
            } else {
                (approx - exact).abs()
            };
            assert!(
                rel_err < 1e-6,
                "Force error too large at angle={angle}: exact={exact}, approx={approx}, rel_err={rel_err}",
            );
        }
    }

    #[test]
    fn test_splined_periodic_dihedral() {
        let pot = PeriodicDihedral::new(0.0, 5.0, 3.0);
        let splined = SplinedDihedral::new(&pot, 1000);

        let test_angles = [-150.0, -90.0, -45.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0];
        for &angle in &test_angles {
            if angle < -PI || angle > PI {
                continue;
            }
            let exact = pot.fourbody_angle_energy(angle);
            let approx = splined.fourbody_angle_energy(angle);
            let rel_err = if exact.abs() > 1e-10 {
                ((approx - exact) / exact).abs()
            } else {
                (approx - exact).abs()
            };
            assert!(
                rel_err < 1e-4,
                "Periodic energy error at angle={angle}: exact={exact}, approx={approx}, rel_err={rel_err}",
            );
        }
    }

    #[test]
    fn test_energy_and_force() {
        let pot = HarmonicDihedral::new(0.5, 100.0);
        let splined = SplinedDihedral::new(&pot, 500);

        let angle = 1.0;
        let (energy, force) = splined.energy_and_force(angle);
        assert_relative_eq!(
            energy,
            splined.fourbody_angle_energy(angle),
            epsilon = 1e-15
        );
        assert_relative_eq!(force, splined.fourbody_angle_force(angle), epsilon = 1e-15);
    }

    #[test]
    fn test_boundary_behavior() {
        let pot = HarmonicDihedral::new(0.0, 10.0);
        let splined = SplinedDihedral::new(&pot, 200);

        assert_relative_eq!(
            splined.fourbody_angle_energy(-PI),
            pot.fourbody_angle_energy(-PI),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            splined.fourbody_angle_energy(PI),
            pot.fourbody_angle_energy(PI),
            epsilon = 1e-10
        );

        let e_at_min = splined.fourbody_angle_energy(-PI);
        let e_below = splined.fourbody_angle_energy(-PI - 0.1);
        assert_relative_eq!(e_at_min, e_below, epsilon = 1e-10);
    }

    #[test]
    fn test_implements_fourbody_trait() {
        let pot = HarmonicDihedral::new(0.0, 10.0);
        let splined = SplinedDihedral::new(&pot, 200);

        let boxed: Box<dyn FourbodyAngleEnergy> = Box::new(splined);
        let energy = boxed.fourbody_angle_energy(1.0);
        let expected = pot.fourbody_angle_energy(1.0);
        assert_relative_eq!(energy, expected, epsilon = 1e-6);
    }
}
