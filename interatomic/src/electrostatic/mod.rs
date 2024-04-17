// Copyright 2023 Björn Stenqvist and Mikael Lund
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

//! # Electrostatic interactions
//!
//! This module contains functions for computing the electrostatic potential;
//! fields; forces; and energies from and between electric multipoles.
//!
//! ## Examples
//! ~~~
//! use interatomic::electrostatic::*;
//! let permittivity = 80.0;
//! let charge = 1.0;
//! let dist = 9.0;
//! let plain = Coulomb::new(permittivity, 16.0, None); // cutoff at 16
//! assert_eq!(plain.ion_potential(charge, dist), charge / dist);
//! ~~~

mod ewald;
pub use ewald::*;
mod coulomb;
pub use coulomb::*;
mod poisson;
use crate::{Matrix3, Vector3};
pub use poisson::*;

/// # Short-range function for electrostatic interaction schemes
///
/// The short-range function, _S(q)_, is a function of the reduced distance _q_ = _r_ / _r<sub>c</sub>_,
/// where _r_ is the distance between the interacting particles and _r<sub>c</sub>_ is the cutoff
/// distance.
/// All _schemes_ implement this trait and is a requirement for the [`Potential`]; [`Field`]; [`Force`]; and [`Energy`] traits.
/// In connection with Ewald summation scemes, the short-range function is also known as the
/// _splitting function_.
/// There it is used to split the electrostatic interaction into a short-range part and a long-range part.
/// The energy between two point charges is,
/// $$ u(r) = \frac{q_1 q_2}{r} \cdot e^{-\kappa r} \cdot S(q) $$
/// and all other quantities are derived from this.
pub trait ShortRangeFunction: crate::Cutoff {
    /// Inverse Debye screening length.
    ///
    /// The default implementation returns `None`.
    fn kappa(&self) -> Option<f64> {
        None
    }
    /// Prefactor in LENGTH / CHARGE^2 * ENERGY units
    ///
    /// For example if the unit of charge is the elementary charge,
    /// letting `prefactor()` return the Bjerrum length, the energy
    /// will be in units of kT.
    fn prefactor(&self) -> f64;
    /// Short-range function.
    fn short_range_f0(&self, q: f64) -> f64;
    /// First derivative of the short-range function.
    fn short_range_f1(&self, q: f64) -> f64;
    /// Second derivative of the short-range function.
    fn short_range_f2(&self, q: f64) -> f64;
    /// Third derivative of the short-range function.
    fn short_range_f3(&self, q: f64) -> f64;
}

/// # Potential from electric multipoles
pub trait MultipolePotential: ShortRangeFunction {
    #[inline]
    fn ion_potential(&self, charge: f64, distance: f64) -> f64 {
        if distance >= self.cutoff() {
            return 0.0;
        }
        let q = distance / self.cutoff();
        charge / distance
            * self.short_range_f0(q)
            * self.kappa().map_or(1.0, |kappa| (-kappa * distance).exp())
    }
    /// Electrostatic potential from a point dipole.
    ///
    /// Parameters:
    /// - `dipole`: Dipole moment, UNIT: [ ( input length ) x ( input charge ) ]
    /// - `r`: Distance vector from the dipole, UNIT: [ input length ]
    ///
    /// Returns:
    /// - Dipole potential, UNIT: [ ( input charge ) / ( input length ) ]
    ///
    /// The potential from a point dipole is described by the formula:
    /// Phi(mu, r) = (mu dot r) / (|r|^2) * [s(q) - q * s'(q)] * exp(-kr)
    fn dipole_potential(&self, dipole: &Vector3, r: &Vector3) -> f64 {
        let r_squared = r.norm_squared();
        if r_squared >= self.cutoff_squared() {
            return 0.0;
        }
        let r1 = r.norm();
        let q = r1 / self.cutoff();
        if let Some(kappa) = self.kappa() {
            let kr = kappa * r1;
            dipole.dot(r) / (r_squared * r1)
                * (self.short_range_f0(q) * (1.0 + kr) - q * self.short_range_f1(q))
                * (-kr).exp()
        } else {
            dipole.dot(r) / (r_squared * r1) * self.short_range_f0(q)
        }
    }

    fn quadrupole_potential(&self, quad: &Matrix3, r: &Vector3) -> f64 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return 0.0;
        }
        let r1 = r.norm();
        let q = r1 / self.cutoff();
        let kr = self.kappa().unwrap_or(0.0) * r1;
        let s0 = self.short_range_f0(q);
        let s1 = self.short_range_f1(q);
        let s2 = self.short_range_f2(q);
        let a =
            s0 * (1.0 + kr + kr * kr / 3.0) - q * s1 * (1.0 + 2.0 / 3.0 * kr) + q * q / 3.0 * s2;
        let b = (s0 * kr * kr - 2.0 * kr * q * s1 + s2 * q * q) / 3.0;
        0.5 * ((3.0 / r2 * (r.transpose() * quad * r)[0] - quad.trace()) * a + quad.trace() * b)
            / (r1 * r2)
            * (-kr).exp()
    }
}

/// # Field due to electric multipoles
pub trait MultipoleField: ShortRangeFunction {
    /// Electrostatic field from point charge.
    ///
    /// Parameters:
    /// - `charge`: Point charge (input charge) [UNIT: input charge]
    /// - `r`: Distance vector from point charge (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Field from charge [UNIT: (input charge) / (input length)^2]
    ///
    /// The field from a charge is described by the formula:
    /// E(z, r) = z * r / |r|^2 * (s(q) - q * s'(q))
    fn ion_field(&self, charge: f64, r: &Vector3) -> Vector3 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return Vector3::zeros();
        }
        let r1 = r.norm();
        let q = r1 / self.cutoff();
        if let Some(kappa) = self.kappa() {
            let kr = kappa * r1;
            charge * r / (r2 * r1)
                * (self.short_range_f0(q) * (1.0 + kr) - q * self.short_range_f1(q))
                * (-kr).exp()
        } else {
            charge * r / (r2 * r1) * self.short_range_f0(q)
        }
    }

    /// Electrostatic field from point dipole.
    ///
    /// Parameters:
    /// - `dipole`: Point dipole (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `r`: Distance vector from point dipole (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Field from dipole [UNIT: (input charge) / (input length)^2]
    ///
    /// The field from a point dipole is described by the formula:
    /// E(mu, r) = (3 * (mu.dot(r) * r / r2) - mu) / r3 *
    ///             (s(q) - q * s'(q) + q^2 / 3 * s''(q)) +
    ///             mu / r3 * (s(q) * kr^2 - 2 * kr * q * s'(q) + q^2 / 3 * s''(q))
    fn dipole_field(&self, dipole: &Vector3, r: &Vector3) -> Vector3 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return Vector3::zeros();
        }
        let r1 = r.norm();
        let r3 = r1 * r2;
        let q = r1 / self.cutoff();
        let kr = self.kappa().unwrap_or(0.0) * r1;
        let kr2 = kr * kr;
        let srf = self.short_range_f0(q);
        let dsrf = self.short_range_f1(q);
        let ddsrf = self.short_range_f2(q);

        let field_d = (3.0 * dipole.dot(r) * r / r2 - dipole) / r3
            * (srf * (1.0 + kr + kr2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr)
                + q * q / 3.0 * ddsrf);

        let field_i = dipole / r3 * (srf * kr2 - 2.0 * kr * q * dsrf + ddsrf * q * q) / 3.0;

        (field_d + field_i) * (-kr).exp()
    }
    /// Electrostatic field from point quadrupole.
    ///
    /// Parameters:
    /// - `quad`: Point quadrupole (input length^2 x input charge) [UNIT: (input length)^2 x (input charge)]
    /// - `r`: Distance vector from point quadrupole (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Field from quadrupole [UNIT: (input charge) / (input length)^2]
    fn quadrupole_field(&self, quad: &Matrix3, r: &Vector3) -> Vector3 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return Vector3::zeros();
        }
        let r_norm = r.norm();
        let r_hat = r / r_norm;
        let q = r_norm / self.cutoff();
        let q2 = q * q;
        let kr = self.kappa().unwrap_or(0.0) * r_norm;
        let kr2 = kr * kr;
        let r4 = r2 * r2;
        let quadrh = quad * r_hat;
        let quad_trh = quad.transpose() * r_hat;

        let quadfactor = (1.0 / r2 * r.transpose() * quad * r)[0]; // 1x1 matrix -> f64 by taking first and only element
        let s0 = self.short_range_f0(q);
        let s1 = self.short_range_f1(q);
        let s2 = self.short_range_f2(q);
        let s3 = self.short_range_f3(q);
        let field_d = 3.0 * ((5.0 * quadfactor - quad.trace()) * r_hat - quadrh - quad_trh) / r4
            * (s0 * (1.0 + kr + kr2 / 3.0) - q * s1 * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * s2);
        let field_i = quadfactor * r_hat / r4
            * (s0 * (1.0 + kr) * kr2 - q * s1 * (3.0 * kr + 2.0) * kr + s2 * (1.0 + 3.0 * kr) * q2
                - q2 * q * s3);
        0.5 * (field_d + field_i) * (-kr).exp()
    }
}

/// # Interaction energy between multipoles
pub trait MultipoleEnergy: MultipolePotential + MultipoleField {
    /// Interaction energy between two point charges
    ///
    /// z1: Point charge, UNIT: [input charge]
    /// z2: Point charge, UNIT: [input charge]
    /// r: Charge-charge separation, UNIT: [input length]
    ///
    /// Returns the interaction energy, UNIT: [(input charge)^2 / (input length)]
    ///
    /// The interaction energy between two charges is described by:
    ///     u(z1, z2, r) = z2 * Phi(z1,r)
    /// where Phi(z1,r) is the potential from ion 1.
    fn ion_ion_energy(&self, charge1: f64, charge2: f64, r: f64) -> f64 {
        charge2 * self.ion_potential(charge1, r)
    }

    /// Interaction energy between a point charge and a point dipole
    ///
    /// - `charge`: Point charge, UNIT: [input charge]
    /// - `dipole`: Dipole moment, UNIT: [(input length) x (input charge)]
    /// - `r`: Distance-vector between dipole and charge, r = r_mu - r_z, UNIT: [input length]
    ///
    /// Returns the interaction energy, UNIT: [(input charge)^2 / (input length)]
    ///
    /// The interaction energy between an ion and a dipole is described by:
    ///     u(z, mu, r) = z * Phi(mu, -r)
    /// where Phi(mu, -r) is the potential from the dipole at the location of the ion.
    /// This interaction can also be described by:
    ///     u(z, mu, r) = -mu.dot(E(z, r))
    /// where E(charge, r) is the field from the ion at the location of the dipole.
    fn ion_dipole_energy(&self, charge: f64, dipole: &Vector3, r: &Vector3) -> f64 {
        // Both expressions below give the same answer. Keep for possible optimization in the future.
        // return -dipole_moment.dot(self.ion_field(charge, r)); // field from charge interacting with dipole
        charge * self.dipole_potential(dipole, &(-r)) // potential of dipole interacting with charge
    }

    /// Interaction energy between two point dipoles
    ///
    /// - `dipole1`: Dipole moment of particle 1, UNIT: [(input length) x (input charge)]
    /// - `dipole2`: Dipole moment of particle 2, UNIT: [(input length) x (input charge)]
    /// r: Distance-vector between dipoles, r = r_mu_2 - r_mu_1, UNIT: [input length]
    ///
    /// Returns the interaction energy, UNIT: [(input charge)^2 / (input length)]
    ///
    /// The interaction energy between two dipoles is described by:
    ///     u(mu1, mu2, r) = -mu1.dot(E(mu2, r))
    /// where E(mu2, r) is the field from dipole 2 at the location of dipole 1.
    fn dipole_dipole_energy(&self, dipole1: &Vector3, dipole2: &Vector3, r: &Vector3) -> f64 {
        -dipole1.dot(&self.dipole_field(dipole2, r))
    }

    /// Interaction energy between a point charge and a point quadrupole
    ///
    /// - `charge`: Point charge, UNIT: [input charge]
    /// - `quad`: Quadrupole moment, UNIT: [(input length)^2 x (input charge)]
    /// - `r`: Distance-vector between quadrupole and charge, r = r_Q - r_z, UNIT: [input length]
    ///
    /// Returns the interaction energy, UNIT: [(input charge)^2 / (input length)]
    ///
    /// The interaction energy between an ion and a quadrupole is described by:
    ///     u(z, Q, r) = z * Phi(Q, -r)
    /// where Phi(Q, -r) is the potential from the quadrupole at the location of the ion.
    fn ion_quadrupole_energy(&self, charge: f64, quad: &Matrix3, r: &Vector3) -> f64 {
        charge * self.quadrupole_potential(quad, &(-r)) // potential of quadrupole interacting with charge
    }
}

/// # Force between multipoles
pub trait MultipoleForce: MultipoleField {
    /// Force between two point charges.
    ///
    /// Parameters:
    /// - `charge1`: Point charge (input charge) [UNIT: input charge]
    /// - `charge2`: Point charge (input charge) [UNIT: input charge]
    /// - `r`: Distance vector between charges (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Interaction force (input charge^2 / input length^2) [UNIT: (input charge)^2 / (input length)^2]
    ///
    /// The force between two point charges is described by the formula:
    /// F(z1, z2, r) = z2 * E(z1, r)
    ///
    /// where:
    /// - `charge1`: Point charge
    /// - `charge2`: Point charge
    /// - `r`: Distance vector between charges
    /// - `E(zA, r)`: Field from ion A at the location of ion B
    fn ion_ion_force(&self, charge1: f64, charge2: f64, r: &Vector3) -> Vector3 {
        charge2 * self.ion_field(charge1, r)
    }
    /// Interaction force between a point charge and a point dipole.
    ///
    /// Parameters:
    /// - `charge`: Charge (input charge) [UNIT: input charge]
    /// - `dipole`: Dipole moment (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `r`: Distance vector between dipole and charge (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Interaction force (input charge^2 / input length^2) [UNIT: (input charge)^2 / (input length)^2]
    ///
    /// The force between an ion and a dipole is described by the formula:
    /// F(charge, mu, r) = charge * E(mu, r)
    ///
    /// where:
    /// - `charge`: Charge
    /// - `mu`: Dipole moment
    /// - `r`: Distance vector between dipole and charge
    /// - `E(mu, r)`: Field from the dipole at the location of the ion
    fn ion_dipole_force(&self, charge: f64, dipole: &Vector3, r: &Vector3) -> Vector3 {
        charge * self.dipole_field(dipole, r)
    }

    /// Interaction force between two point dipoles.
    ///
    /// Parameters:
    /// - `mu1`: Dipole moment of particle 1 (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `mu2`: Dipole moment of particle 2 (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `r`: Distance vector between dipoles (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Interaction force (input charge^2 / input length^2) [UNIT: (input charge)^2 / (input length)^2]
    ///
    /// The force between two dipoles is described by the formula:
    /// F(mu1, mu2, r) = FD(mu1, mu2, r) * (s(q) - q * s'(q) + (q^2 / 3) * s''(q))
    ///                  + FI(mu1, mu2, r) * (s''(q) - q * s'''(q)) * q^2 * exp(-kr)
    fn dipole_dipole_force(&self, mu1: &Vector3, mu2: &Vector3, r: &Vector3) -> Vector3 {
        let r2 = r.norm_squared();
        if r2 < self.cutoff_squared() {
            let r1 = r.norm();
            let rh = r / r1;
            let q = r1 / self.cutoff();
            let q2 = q * q;
            let kr = self.kappa().unwrap_or(0.0) * r1;
            let r4 = r2 * r2;
            let mu1_dot_rh = mu1.dot(&rh);
            let mu2_dot_rh = mu2.dot(&rh);
            let mut force_d = 3.0
                * ((5.0 * mu1_dot_rh * mu2_dot_rh - mu1.dot(mu2)) * rh
                    - mu2_dot_rh * mu1
                    - mu1_dot_rh * mu2)
                / r4;
            let srf = self.short_range_f0(q);
            let dsrf = self.short_range_f1(q);
            let ddsrf = self.short_range_f2(q);
            let dddsrf = self.short_range_f3(q);
            force_d *= srf * (1.0 + kr + kr * kr / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr)
                + q2 / 3.0 * ddsrf;
            let force_i = mu1_dot_rh * mu2_dot_rh * rh / r4
                * (srf * (1.0 + kr) * kr * kr - q * dsrf * (3.0 * kr + 2.0) * kr
                    + ddsrf * (1.0 + 3.0 * kr) * q2
                    - q2 * q * dddsrf);
            (force_d + force_i) * (-kr).exp()
        } else {
            Vector3::zeros()
        }
    }

    /**
     * Interaction force between a point charge and a point quadrupole.
     *
     * Parameters:
     * - `charge`: Point charge (input charge) [UNIT: input charge]
     * - `quad`: Point quadrupole (input length^2 x input charge) [UNIT: (input length)^2 x (input charge)]
     * - `r`: Distance vector between particles (input length) [UNIT: input length]
     *
     * Returns:
     * Interaction force [UNIT: (input charge)^2 / (input length)^2]
     *
     * The force between a point charge and a point quadrupole is described by the formula:
     * F(charge, quad, r) = charge * E(quad, r)
     * where E(quad, r) is the field from the quadrupole at the location of the ion.
     */
    fn ion_quadrupole_force(&self, charge: f64, quad: Matrix3, r: Vector3) -> Vector3 {
        charge * self.quadrupole_field(&quad, &r)
    }
}

/// Test electric constant
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    #[test]
    fn test_electric_constant() {
        let bjerrum_length = 7.1; // angstrom
        let rel_dielectric_const = 80.0;
        assert_relative_eq!(
            crate::ELECTRIC_PREFACTOR / rel_dielectric_const / bjerrum_length,
            2.4460467895137676 // In kJ/mol, roughly 1 KT at room temperature
        );
    }
}
