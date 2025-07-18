// Copyright 2023 Mikael Lund
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

//! # Electric multipole interactions
//!
//! This module provides tools for calculating the two-body interaction energy between
//! electric multipole moments, such as monopoles, dipoles, quadrupoles etc.

use crate::{twobody::IsotropicTwobodyEnergy, Vector3};
use coulomb::{
    pairwise::MultipoleEnergy,
    permittivity::{ConstantPermittivity, RelativePermittivity},
};
#[cfg(feature = "serde")]
use serde::Serialize;
use std::fmt::{Debug, Display};

/// Monopole-monopole interaction energy
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct IonIon<T: MultipoleEnergy> {
    /// Charge number product of the two particles, z₁ × z₂
    #[cfg_attr(feature = "serde", serde(rename = "z₁z₂", alias = "z1z2"))]
    charge_product: f64,
    /// Potential energy function to use.
    #[cfg_attr(feature = "serde", serde(skip))]
    scheme: T,
    /// Relative dielectric constant of the medium
    permittivity: ConstantPermittivity,
}

impl<T: MultipoleEnergy + Display> Display for IonIon<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}, {}", self.scheme, self.permittivity)
    }
}

impl<T: MultipoleEnergy> coulomb::DebyeLength for IonIon<T> {
    fn kappa(&self) -> Option<f64> {
        self.scheme.kappa()
    }
    fn set_debye_length(&mut self, debye_length: Option<f64>) -> anyhow::Result<()> {
        self.scheme.set_debye_length(debye_length)
    }
    fn debye_length(&self) -> Option<f64> {
        self.scheme.debye_length()
    }
}

impl<T: MultipoleEnergy> RelativePermittivity for IonIon<T> {
    fn permittivity(&self, _temperature: f64) -> anyhow::Result<f64> {
        Ok(f64::from(self.permittivity))
    }
    fn set_permittivity(&mut self, permittivity: f64) -> anyhow::Result<()> {
        self.permittivity = ConstantPermittivity::new(permittivity);
        Ok(())
    }
}

impl<T: MultipoleEnergy> IonIon<T> {
    /// Create a new ion-ion interaction
    pub const fn new(charge_product: f64, permittivity: ConstantPermittivity, scheme: T) -> Self {
        Self {
            charge_product,
            permittivity,
            scheme,
        }
    }
}

impl<T: MultipoleEnergy + Debug + Clone + PartialEq + Send + Sync> IsotropicTwobodyEnergy
    for IonIon<T>
{
    /// Calculate the isotropic twobody energy (kJ/mol)
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        coulomb::TO_CHEMISTRY_UNIT / f64::from(self.permittivity)
            * self
                .scheme
                .ion_ion_energy(self.charge_product, 1.0, distance_squared.sqrt())
    }
}

/// Alias for ion-ion with Yukawa
pub type IonIonYukawa<'a> = IonIon<coulomb::pairwise::Yukawa>;

/// Alias for ion-ion with a plain Coulomb potential that can be screened
pub type IonIonPlain<'a> = IonIon<coulomb::pairwise::Plain>;

/// Ion-ion interaction with added ion-induced dipole energy
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct IonIonPolar<T: MultipoleEnergy> {
    pub ionion: IonIon<T>,
    /// Charges of two particles
    charges: (f64, f64),
    /// Common excess polarizabilities of the ion in Å³
    polarizabilities: (f64, f64),
}

impl<T: MultipoleEnergy> IonIonPolar<T> {
    /// Create a new ion-ion interaction with polarizability
    pub const fn new(ionion: IonIon<T>, charges: (f64, f64), polarizabilities: (f64, f64)) -> Self {
        Self {
            ionion,
            charges,
            polarizabilities,
        }
    }
}

impl<T: MultipoleEnergy + Debug + Clone + PartialEq + Send + Sync> IsotropicTwobodyEnergy
    for IonIonPolar<T>
{
    /// Calculate the isotropic twobody energy (kJ/mol)
    ///
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        let r = distance_squared.sqrt();
        let scheme = &self.ionion.scheme;
        let ion_ion = scheme.ion_ion_energy(self.ionion.charge_product, 1.0, r);

        let r = Vector3::new(r, 0.0, 0.0);
        // These terms are always <= 0
        // TODO: This could be optimized by calling `ion_induced_dipole_energy` only once
        // with e.g. alpha=1 and charge=`q1*a0 + q0*a1`.
        let ion_induced_dipole =
            scheme.ion_induced_dipole_energy(self.charges.0, self.polarizabilities.1, &r)
                + scheme.ion_induced_dipole_energy(self.charges.1, self.polarizabilities.0, &-r);

        let to_kjmol = coulomb::TO_CHEMISTRY_UNIT / f64::from(self.ionion.permittivity);
        to_kjmol * (ion_ion + ion_induced_dipole)
    }
}

impl<T: MultipoleEnergy + Display> Display for IonIonPolar<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IonIonPolar({}, {:?})",
            self.ionion, self.polarizabilities
        )
    }
}

// Test ion-ion energy
#[cfg(test)]
mod tests {
    use core::f64;

    use approx::assert_relative_eq;

    use super::*;
    use coulomb::{pairwise::Plain, DebyeLength};

    #[test]
    fn test_ion_ion() {
        let permittivity = ConstantPermittivity::new(80.0);
        let r: f64 = 7.0;
        let cutoff = f64::INFINITY;
        let scheme = Plain::new(cutoff, None);
        let ionion = IonIon::new(1.0, permittivity, scheme);
        let unscreened_energy = ionion.isotropic_twobody_energy(r.powi(2));
        assert_relative_eq!(unscreened_energy, 2.48099031507825);
        let debye_length = 30.0;
        let scheme = Plain::new(cutoff, Some(debye_length));
        let ionion = IonIon::new(1.0, permittivity, scheme);
        let screened_energy = ionion.isotropic_twobody_energy(r.powi(2));
        assert_relative_eq!(
            screened_energy,
            unscreened_energy * (-r / debye_length).exp()
        );
    }
    #[test]
    fn test_ion_ion_polar() {
        let permittivity = ConstantPermittivity::new(80.0);
        let r: f64 = 8.0;
        let charges = (1.0, -1.0);
        let cutoff = f64::INFINITY;
        let polarizabilities = (100.0, 80.0); // Excess polarizability in Å³

        // No salt screening
        let scheme = Plain::new(cutoff, None);
        let ionion = IonIon::new(charges.0 * charges.1, permittivity, scheme);
        let ionion_polar = IonIonPolar::new(ionion.clone(), charges, polarizabilities);
        let energy = ionion_polar.isotropic_twobody_energy(r.powi(2));
        assert_relative_eq!(energy, -2.5524641571630236);

        let induced_energy = energy - ionion.isotropic_twobody_energy(r.powi(2));
        assert_relative_eq!(induced_energy, -0.38159763146955505);
        assert_relative_eq!(
            induced_energy,
            -(100.0 + 80.0) * coulomb::TO_CHEMISTRY_UNIT
                / (2.0 * f64::from(permittivity) * r.powi(4))
        );

        // With salt screening
        let scheme = Plain::new(cutoff, Some(30.0));
        let ionion = IonIon::new(charges.0 * charges.1, permittivity, scheme.clone());
        let ionion_polar = IonIonPolar::new(ionion.clone(), charges, polarizabilities);
        let energy = ionion_polar.isotropic_twobody_energy(r.powi(2));
        assert_relative_eq!(energy, -2.021903629249571);

        let induced_energy = energy - ionion.isotropic_twobody_energy(r.powi(2));
        let kappa = 1.0 / scheme.debye_length().unwrap();
        assert_relative_eq!(induced_energy, -0.3591754384137349);
        assert_relative_eq!(
            induced_energy,
            -(polarizabilities.0 + polarizabilities.1) * coulomb::TO_CHEMISTRY_UNIT
                / (2.0 * f64::from(permittivity))
                * (f64::exp(-r * kappa) * (1.0 / r.powi(2) + kappa / r)).powi(2),
            epsilon = 1e-7
        );
        assert_relative_eq!(
            induced_energy,
            -(polarizabilities.0 + polarizabilities.1) * coulomb::TO_CHEMISTRY_UNIT
                / (2.0 * f64::from(permittivity))
                * f64::exp(-2.0 * r * kappa)
                * (1.0 / r.powi(2) + kappa / r).powi(2),
            epsilon = 1e-7
        );
    }
}
