// Copyright 2023-2024 Mikael Lund
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

//! Implementation of the harmonic dihedral.

use super::FourbodyAngleEnergy;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Harmonic dihedral potential.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct HarmonicDihedral {
    #[cfg_attr(feature = "serde", serde(rename = "aeq"))]
    eq_angle: f64,
    #[cfg_attr(feature = "serde", serde(rename = "k"))]
    spring_constant: f64,
}

impl HarmonicDihedral {
    /// Create a new harmonic dihedral potential with equilibrium angle and spring constant.
    pub const fn new(eq_angle: f64, spring_constant: f64) -> Self {
        Self {
            eq_angle,
            spring_constant,
        }
    }

    /// Equilibrium dihedral angle.
    pub const fn eq_angle(&self) -> f64 {
        self.eq_angle
    }

    /// Spring constant.
    pub const fn spring_constant(&self) -> f64 {
        self.spring_constant
    }
}

impl FourbodyAngleEnergy for HarmonicDihedral {
    #[inline(always)]
    fn fourbody_angle_energy(&self, angle: f64) -> f64 {
        0.5 * self.spring_constant * (angle - self.eq_angle).powi(2)
    }

    #[inline(always)]
    fn fourbody_angle_force(&self, angle: f64) -> f64 {
        -self.spring_constant * (angle - self.eq_angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_6};

    #[test]
    fn test_harmonic_dihedral_force() {
        let dihedral = HarmonicDihedral::new(FRAC_PI_2, 100.0);
        // Energy at 120° in radians
        let phi = 2.0 * std::f64::consts::PI / 3.0;
        assert_relative_eq!(
            dihedral.fourbody_angle_energy(phi),
            13.7077838904,
            epsilon = 1e-6
        );
        // Force = -k(φ - φ_eq) = -100*(2π/3 - π/2) = -100*π/6
        assert_relative_eq!(
            dihedral.fourbody_angle_force(phi),
            -100.0 * FRAC_PI_6,
            epsilon = 1e-6
        );
        // At equilibrium: force = 0
        assert_relative_eq!(
            dihedral.fourbody_angle_force(FRAC_PI_2),
            0.0,
            epsilon = 1e-10
        );
    }
}
