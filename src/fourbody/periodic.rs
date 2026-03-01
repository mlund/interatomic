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

//! Implementation of the periodic dihedral potential.

use super::FourbodyAngleEnergy;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Periodic dihedral potential.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct PeriodicDihedral {
    #[cfg_attr(feature = "serde", serde(rename = "phi"))]
    phase_angle: f64,
    #[cfg_attr(feature = "serde", serde(rename = "k"))]
    spring_constant: f64,
    #[cfg_attr(feature = "serde", serde(rename = "n"))]
    periodicity: f64,
}

impl PeriodicDihedral {
    /// Create a new periodic dihedral potential with phase angle, spring constant, and periodicity.
    pub const fn new(phase_angle: f64, spring_constant: f64, periodicity: f64) -> Self {
        Self {
            phase_angle,
            spring_constant,
            periodicity,
        }
    }
}

impl FourbodyAngleEnergy for PeriodicDihedral {
    /// Energy: k * (1 + cos(n*θ - φ)), where θ and φ are in degrees.
    #[inline(always)]
    fn fourbody_angle_energy(&self, angle: f64) -> f64 {
        let angle_rad = angle.to_radians();
        let phase_rad = self.phase_angle.to_radians();
        self.spring_constant * (1.0 + (self.periodicity * angle_rad - phase_rad).cos())
    }

    /// Force: -dU/dφ (in degrees). Returns k*n*sin(n*θ - φ₀) * π/180.
    #[inline(always)]
    fn fourbody_angle_force(&self, angle: f64) -> f64 {
        let angle_rad = angle.to_radians();
        let phase_rad = self.phase_angle.to_radians();
        self.spring_constant
            * self.periodicity
            * (self.periodicity * angle_rad - phase_rad).sin()
            * std::f64::consts::PI
            / 180.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn periodic_dihedral_energy() {
        // k=10, n=2, phi=180°, angle=0° → 10*(1+cos(0-π)) = 0
        let pot = PeriodicDihedral::new(180.0, 10.0, 2.0);
        assert_relative_eq!(pot.fourbody_angle_energy(0.0), 0.0, epsilon = 1e-10);

        // k=10, n=2, phi=180°, angle=90° → 10*(1+cos(π-π)) = 20
        assert_relative_eq!(pot.fourbody_angle_energy(90.0), 20.0, epsilon = 1e-10);

        // k=5, n=3, phi=0°, angle=60° → 5*(1+cos(π)) = 0
        let pot2 = PeriodicDihedral::new(0.0, 5.0, 3.0);
        assert_relative_eq!(pot2.fourbody_angle_energy(60.0), 0.0, epsilon = 1e-10);

        // k=5, n=3, phi=0°, angle=0° → 5*(1+cos(0)) = 10
        assert_relative_eq!(pot2.fourbody_angle_energy(0.0), 10.0, epsilon = 1e-10);

        // k=1, n=1, phi=0°, angle=180° → 1*(1+cos(π)) = 0
        let pot3 = PeriodicDihedral::new(0.0, 1.0, 1.0);
        assert_relative_eq!(pot3.fourbody_angle_energy(180.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_periodic_dihedral_force() {
        let pot = PeriodicDihedral::new(0.0, 100.0, 3.0);
        // Energy cross-checks
        assert_relative_eq!(pot.fourbody_angle_energy(120.0), 200.0, epsilon = 1e-10);
        assert_relative_eq!(pot.fourbody_angle_energy(60.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(pot.fourbody_angle_energy(90.0), 100.0, epsilon = 1e-10);

        // At φ=90°: -dU/dφ_deg = kn*sin(270°)*π/180 = -300*π/180
        assert_relative_eq!(
            pot.fourbody_angle_force(90.0),
            -300.0 * std::f64::consts::PI / 180.0,
            epsilon = 1e-10
        );
        // At φ=60° and φ=120°: sin(nφ)=sin(180°)=0 and sin(360°)=0 → force=0
        assert_relative_eq!(pot.fourbody_angle_force(60.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(pot.fourbody_angle_force(120.0), 0.0, epsilon = 1e-10);
    }
}
