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

//! Implementation of the three-body harmonic potential.

use super::ThreebodyAngleEnergy;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Harmonic torsion potential.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct HarmonicTorsion {
    #[cfg_attr(feature = "serde", serde(rename = "aeq"))]
    eq_angle: f64,
    #[cfg_attr(feature = "serde", serde(rename = "k"))]
    spring_constant: f64,
}

impl HarmonicTorsion {
    /// Create a new harmonic torsion potential with equilibrium angle and spring constant.
    pub const fn new(eq_angle: f64, spring_constant: f64) -> Self {
        Self {
            eq_angle,
            spring_constant,
        }
    }
}

impl ThreebodyAngleEnergy for HarmonicTorsion {
    #[inline(always)]
    fn threebody_angle_energy(&self, angle: f64) -> f64 {
        0.5 * self.spring_constant * (angle - self.eq_angle).powi(2)
    }

    #[inline(always)]
    fn threebody_angle_force(&self, angle: f64) -> f64 {
        -self.spring_constant * (angle - self.eq_angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    #[test]
    fn test_harmonic_torsion_force() {
        let torsion = HarmonicTorsion::new(FRAC_PI_4, 1.0);
        // At θ=π/2: force = -1*(π/2 - π/4) = -π/4
        assert_relative_eq!(
            torsion.threebody_angle_force(FRAC_PI_2),
            -FRAC_PI_4,
            epsilon = 1e-10
        );
        // At equilibrium: force = 0
        assert_relative_eq!(
            torsion.threebody_angle_force(FRAC_PI_4),
            0.0,
            epsilon = 1e-10
        );
    }
}
