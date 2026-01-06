// Copyright 2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Kim-Hummer coarse-grained protein potential.
//!
//! References:
//! - Kim & Hummer (2008) J. Mol. Biol. 375, 1416-1433
//!   <https://doi.org/10.1016/j.jmb.2007.11.063>
//! - Miyazawa & Jernigan (1996) J. Mol. Biol. 256, 623-644
//!   <https://doi.org/10.1006/jmbi.1996.0114>

use crate::twobody::IsotropicTwobodyEnergy;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Kim-Hummer coarse-grained protein potential
///
/// A Lennard-Jones-like potential with modified behavior based on the sign of epsilon:
///
/// - **Attractive (ε < 0)**: Standard LJ potential
///   $$u(r) = 4|\varepsilon|\left[(\sigma/r)^{12} - (\sigma/r)^6\right]$$
///
/// - **Repulsive (ε > 0)**: Two branches joined at r₀ = 2^(1/6)σ
///   - Inner (r < r₀): $u(r) = 4\varepsilon\left[(\sigma/r)^{12} - (\sigma/r)^6\right] + 2\varepsilon$
///   - Outer (r ≥ r₀): $u(r) = -4\varepsilon\left[(\sigma/r)^{12} - (\sigma/r)^6\right]$
///
/// - **Neutral (ε = 0)**: Soft repulsive wall
///   $$u(r) = 0.01(\sigma/r)^{12}$$
///
/// The potential is continuous at r₀ for all epsilon values.
///
/// # Examples
/// ```
/// use interatomic::twobody::*;
/// let kh = KimHummer::new(-0.5, 6.0);
/// // At r = 2^(1/6) * sigma, attractive potential has minimum = epsilon
/// let r_min = f64::powf(2.0, 1.0 / 6.0) * 6.0;
/// let u = kh.isotropic_twobody_energy(r_min * r_min);
/// assert!((u - (-0.5)).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct KimHummer {
    /// Interaction strength (kT)
    #[cfg_attr(feature = "serde", serde(alias = "eps", alias = "ε"))]
    pub epsilon: f64,
    /// Contact distance (Å or other length unit)
    #[cfg_attr(feature = "serde", serde(alias = "σ"))]
    pub sigma: f64,
}

impl KimHummer {
    /// 2^(2/6) = 2^(1/3)
    const TWO_TO_TWO_SIXTH: f64 = 1.2599210498948732;

    /// Create a new Kim-Hummer potential.
    ///
    /// # Arguments
    /// * `epsilon` - Interaction strength (kT). Negative for attractive, positive for repulsive.
    /// * `sigma` - Contact distance (Å)
    pub const fn new(epsilon: f64, sigma: f64) -> Self {
        Self { epsilon, sigma }
    }

    /// Returns the position of the potential cusp/minimum: r₀ = 2^(1/6) * σ
    #[inline(always)]
    pub fn r0(&self) -> f64 {
        f64::powf(2.0, 1.0 / 6.0) * self.sigma
    }

    /// Returns r₀² = 2^(1/3) * σ²
    #[inline(always)]
    fn r0_squared(&self) -> f64 {
        Self::TWO_TO_TWO_SIXTH * self.sigma * self.sigma
    }
}

impl IsotropicTwobodyEnergy for KimHummer {
    #[inline]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        let sigma_squared = self.sigma * self.sigma;
        let sr6 = (sigma_squared / distance_squared).powi(3); // (σ/r)^6
        let sr12 = sr6 * sr6; // (σ/r)^12

        if self.epsilon < 0.0 {
            // Attractive: standard LJ with well depth |ε|
            4.0 * self.epsilon.abs() * (sr12 - sr6)
        } else if self.epsilon > 0.0 {
            // Repulsive: two branches
            let r0_squared = self.r0_squared();
            if distance_squared < r0_squared {
                // Inner branch: standard LJ shifted up by 2ε
                4.0 * self.epsilon * (sr12 - sr6) + 2.0 * self.epsilon
            } else {
                // Outer branch: inverted LJ (always positive, decaying to zero)
                -4.0 * self.epsilon * (sr12 - sr6)
            }
        } else {
            // Neutral (ε = 0): soft wall
            0.01 * sr12
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const SIGMA: f64 = 6.0;

    fn r0() -> f64 {
        f64::powf(2.0, 1.0 / 6.0) * SIGMA
    }

    // --- Attractive pairs (ε < 0) ---

    #[test]
    fn test_attractive_minimum_value() {
        // At r = 2^(1/6)σ, U should equal ε for attractive pairs
        let epsilon = -0.5;
        let kh = KimHummer::new(epsilon, SIGMA);
        let r = r0();
        let u = kh.isotropic_twobody_energy(r * r);
        assert_relative_eq!(u, epsilon, epsilon = 1e-8);
    }

    #[test]
    fn test_attractive_at_sigma() {
        // At r = σ, U should be 0 for attractive LJ
        let epsilon = -0.5;
        let kh = KimHummer::new(epsilon, SIGMA);
        let r_squared = SIGMA * SIGMA;
        let u = kh.isotropic_twobody_energy(r_squared);
        assert_relative_eq!(u, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_attractive_repulsive_core() {
        // For r < σ, U should be positive (repulsive core)
        let epsilon = -0.5;
        let kh = KimHummer::new(epsilon, SIGMA);
        let r = 0.9 * SIGMA;
        let u = kh.isotropic_twobody_energy(r * r);
        assert!(u > 0.0);
    }

    #[test]
    fn test_attractive_well_shape() {
        // U should be negative between σ and large r for attractive pairs
        let epsilon = -0.5;
        let kh = KimHummer::new(epsilon, SIGMA);
        let r = 1.5 * SIGMA;
        let u = kh.isotropic_twobody_energy(r * r);
        assert!(u < 0.0);
        assert!(u > epsilon); // Above well minimum
    }

    // --- Repulsive pairs (ε > 0) ---

    #[test]
    fn test_repulsive_cusp_value() {
        // At r = 2^(1/6)σ, U should equal ε for repulsive pairs
        let epsilon = 0.3;
        let kh = KimHummer::new(epsilon, SIGMA);
        let r = r0();
        let u = kh.isotropic_twobody_energy(r * r);
        assert_relative_eq!(u, epsilon, epsilon = 1e-8);
    }

    #[test]
    fn test_repulsive_always_positive() {
        // Repulsive potential should be positive everywhere
        let epsilon = 0.3;
        let kh = KimHummer::new(epsilon, SIGMA);
        for i in 0..100 {
            let r = 0.8 * SIGMA + (i as f64) * 0.022 * SIGMA; // 0.8σ to ~3σ
            let u = kh.isotropic_twobody_energy(r * r);
            assert!(u > 0.0, "U({}) = {} should be > 0", r, u);
        }
    }

    #[test]
    fn test_repulsive_inner_branch() {
        // Test inner branch (r < r0): U = 4ε[(σ/r)^12 - (σ/r)^6] + 2ε
        let epsilon = 0.3;
        let kh = KimHummer::new(epsilon, SIGMA);
        let r = 0.95 * r0(); // Just inside r0
        let sr6 = (SIGMA / r).powi(6);
        let sr12 = sr6 * sr6;
        let expected = 4.0 * epsilon * (sr12 - sr6) + 2.0 * epsilon;
        let u = kh.isotropic_twobody_energy(r * r);
        assert_relative_eq!(u, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_repulsive_outer_branch() {
        // Test outer branch (r >= r0): U = -4ε[(σ/r)^12 - (σ/r)^6]
        let epsilon = 0.3;
        let kh = KimHummer::new(epsilon, SIGMA);
        let r = 1.5 * SIGMA; // Outside r0
        let sr6 = (SIGMA / r).powi(6);
        let sr12 = sr6 * sr6;
        let expected = -4.0 * epsilon * (sr12 - sr6);
        let u = kh.isotropic_twobody_energy(r * r);
        assert_relative_eq!(u, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_repulsive_continuity_at_r0() {
        // Potential should be continuous at r = r0
        let epsilon = 0.3;
        let kh = KimHummer::new(epsilon, SIGMA);
        let delta = 1e-8;
        let r0_val = r0();
        let r_below = r0_val - delta;
        let r_above = r0_val + delta;
        let u_below = kh.isotropic_twobody_energy(r_below * r_below);
        let u_above = kh.isotropic_twobody_energy(r_above * r_above);
        assert_relative_eq!(u_below, u_above, epsilon = 1e-4);
    }

    // --- Neutral pairs (ε = 0) ---

    #[test]
    fn test_neutral_soft_wall() {
        // For ε = 0, U = 0.01(σ/r)^12
        let epsilon = 0.0;
        let kh = KimHummer::new(epsilon, SIGMA);
        let r = SIGMA;
        let expected = 0.01 * (SIGMA / r).powi(12);
        let u = kh.isotropic_twobody_energy(r * r);
        assert_relative_eq!(u, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_neutral_always_positive() {
        // Neutral potential should be positive everywhere
        let epsilon = 0.0;
        let kh = KimHummer::new(epsilon, SIGMA);
        for i in 0..100 {
            let r = 0.8 * SIGMA + (i as f64) * 0.022 * SIGMA;
            let u = kh.isotropic_twobody_energy(r * r);
            assert!(u > 0.0, "U({}) = {} should be > 0", r, u);
        }
    }

    #[test]
    fn test_neutral_monotonic_decrease() {
        // Neutral potential should decrease monotonically with r
        let epsilon = 0.0;
        let kh = KimHummer::new(epsilon, SIGMA);
        let mut prev_u = f64::INFINITY;
        for i in 1..100 {
            let r = 0.8 * SIGMA + (i as f64) * 0.022 * SIGMA;
            let u = kh.isotropic_twobody_energy(r * r);
            assert!(u < prev_u, "U({}) should be < U(prev)", r);
            prev_u = u;
        }
    }

    // --- Edge cases ---

    #[test]
    fn test_large_r_approaches_zero() {
        // Potential should approach 0 at large r
        for epsilon in [-0.5, 0.0, 0.3] {
            let kh = KimHummer::new(epsilon, SIGMA);
            let r = 100.0 * SIGMA;
            let u = kh.isotropic_twobody_energy(r * r);
            assert_relative_eq!(u, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_r_zero_is_nan() {
        // At r = 0, potential is NaN (inf - inf), consistent with LennardJones
        for epsilon in [-0.5, 0.0, 0.3] {
            let kh = KimHummer::new(epsilon, SIGMA);
            let u = kh.isotropic_twobody_energy(0.0);
            assert!(u.is_nan() || u.is_infinite());
        }
    }
}
