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

//! ## Twobody interactions
//!
//! Module for describing exactly two particles interacting with each other.

use crate::Cutoff;
pub use crate::Vector3;
use core::fmt::Debug;
use core::iter::Sum;
use core::ops::Add;
use dyn_clone::DynClone;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::sync::Arc;

mod ashbaugh_hatch;
#[cfg(feature = "custom")]
mod custom;
mod fene;
mod hardsphere;
mod harmonic;
mod hermite;
mod kimhummer;
mod lennard_jones;
mod mie;
mod morse;
mod multipole;
pub mod potential;
mod ureybradley;
mod wca;
pub use ashbaugh_hatch::AshbaughHatch;
#[cfg(feature = "custom")]
pub use custom::{CustomPotential, CustomPotentialError};
pub use fene::FENE;
pub use hardsphere::HardSphere;
pub use harmonic::Harmonic;
pub(crate) use hermite::compute_uniform_hermite_coeffs;
#[cfg(feature = "simd")]
pub use hermite::{
    simd_f32_from_array, simd_f32_to_array, SimdArrayF32, SimdF32, SplineTableSimd,
    SplineTableSimdF32, LANES_F32,
};
pub use hermite::{
    spline_potential, GridTrim, GridType, SplineCoeffs, SplineConfig, SplineError, SplineStats,
    SplinedPotential, ValidationResult,
};
pub use kimhummer::KimHummer;
pub use lennard_jones::LennardJones;
pub use mie::Mie;
pub use morse::Morse;
pub use multipole::{IonIon, IonIonPlain, IonIonPolar, IonIonYukawa};
pub use ureybradley::UreyBradley;
pub use wca::WeeksChandlerAndersen;

/// Relative orientation between a pair of anisotropic particles.
///
/// # Todo
/// Unfinished and still not decided how to implement
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RelativeOrientation {
    /// Distance between the two particles
    pub distance: Vector3,
    /// Orientation vector of the particles
    pub orientation: Vector3,
}

/// Potential energy between a pair of anisotropic particles.
pub trait AnisotropicTwobodyEnergy: Send + Sync {
    /// Interaction energy between a pair of anisotropic particles, 𝑈(𝒓).
    fn anisotropic_twobody_energy(&self, orientation: &RelativeOrientation) -> f64;

    /// Force magnitude due to an anisotropic interaction potential, 𝐹(𝒓) = -𝞩𝑈(𝒓)
    fn anisotropic_twobody_force(&self, _: &RelativeOrientation) -> Vector3 {
        todo!()
    }
}

/// Potential energy between a pair of isotropic particles, 𝑈(𝑟).
pub trait IsotropicTwobodyEnergy: AnisotropicTwobodyEnergy + DynClone + Debug + Cutoff {
    /// Interaction energy between a pair of isotropic particles.
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64;

    /// Negative derivative of the potential with respect to squared distance, −∂𝑈/∂(𝑟²).
    ///
    /// Related to the force magnitude by 𝐹(𝑟) = −d𝑈/d𝑟 = 2𝑟 · (−d𝑈/d(𝑟²)).
    /// The default implementation uses a central difference and should be overridden
    /// with the exact analytical expression for better speed and accuracy.
    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        const EPS: f64 = 1e-6;
        let delta_u = self.isotropic_twobody_energy(distance_squared + EPS)
            - self.isotropic_twobody_energy(distance_squared - EPS);
        -delta_u / (2.0 * EPS)
    }
}

dyn_clone::clone_trait_object!(IsotropicTwobodyEnergy);

/// Calculate the squared norm of a Vector3.
fn norm_squared(v: &Vector3) -> f64 {
    v.x * v.x + v.y * v.y + v.z * v.z
}

/// All isotropic potentials implement the anisotropic trait.
impl<T: IsotropicTwobodyEnergy> AnisotropicTwobodyEnergy for T {
    fn anisotropic_twobody_energy(&self, orientation: &RelativeOrientation) -> f64 {
        self.isotropic_twobody_energy(norm_squared(&orientation.distance))
    }
    fn anisotropic_twobody_force(&self, orientation: &RelativeOrientation) -> Vector3 {
        let r_squared = norm_squared(&orientation.distance);
        let inv_r = 1.0 / r_squared.sqrt();
        let f = self.isotropic_twobody_force(r_squared) * inv_r;
        Vector3 {
            x: f * orientation.distance.x,
            y: f * orientation.distance.y,
            z: f * orientation.distance.z,
        }
    }
}

/// A truncation distance that can't bound an interaction: not finite, or not positive.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("cutoff must be finite and > 0, got {0}")]
pub struct InvalidCutoff(pub f64);

impl InvalidCutoff {
    /// Accept only a finite, positive distance.
    pub fn check(cutoff: f64) -> Result<f64, Self> {
        if cutoff.is_finite() && cutoff > 0.0 {
            Ok(cutoff)
        } else {
            Err(Self(cutoff))
        }
    }
}

/// Structure representing an interaction with always zero energy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoInteraction;

impl Cutoff for NoInteraction {
    /// Zero range: the identity of the sum rule (max), so empty folds add nothing.
    fn cutoff(&self) -> f64 {
        0.0
    }
}

impl IsotropicTwobodyEnergy for NoInteraction {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, _distance_squared: f64) -> f64 {
        0.0
    }

    #[inline(always)]
    fn isotropic_twobody_force(&self, _distance_squared: f64) -> f64 {
        0.0
    }
}

/// Combine two twobody pair potentials.
///
/// This works with both static and dynamic dispatch.
/// For dynamic dispatch, `Box<dyn IsotropicTwobodyEnergy>`
/// can be aggregated using the `+` operator.
///
/// Its `cutoff()` is the larger of the two: the sum interacts wherever either term does.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Combined<T, U>(T, U);

impl<T: IsotropicTwobodyEnergy, U: IsotropicTwobodyEnergy> Combined<T, U> {
    /// Create a new combined potential from two isotropic twobody potentials.
    pub const fn new(t: T, u: U) -> Self {
        Self(t, u)
    }
}

impl<T: IsotropicTwobodyEnergy + Clone, U: IsotropicTwobodyEnergy + Clone> IsotropicTwobodyEnergy
    for Combined<T, U>
{
    #[inline(always)]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        self.0.isotropic_twobody_energy(distance_squared)
            + self.1.isotropic_twobody_energy(distance_squared)
    }

    #[inline(always)]
    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        self.0.isotropic_twobody_force(distance_squared)
            + self.1.isotropic_twobody_force(distance_squared)
    }
}

impl<T: Cutoff, U: Cutoff> Cutoff for Combined<T, U> {
    fn cutoff(&self) -> f64 {
        self.0.cutoff().max(self.1.cutoff())
    }
    fn lower_cutoff(&self) -> f64 {
        self.0.lower_cutoff().max(self.1.lower_cutoff())
    }
}

impl Cutoff for Box<dyn IsotropicTwobodyEnergy> {
    fn cutoff(&self) -> f64 {
        self.as_ref().cutoff()
    }
    fn lower_cutoff(&self) -> f64 {
        self.as_ref().lower_cutoff()
    }
}

impl IsotropicTwobodyEnergy for Box<dyn IsotropicTwobodyEnergy> {
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        self.as_ref().isotropic_twobody_energy(distance_squared)
    }

    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        self.as_ref().isotropic_twobody_force(distance_squared)
    }
}

/// Thread-safe shared wrapper for dynamic isotropic potentials.
///
/// This is a newtype wrapper around `Arc<dyn IsotropicTwobodyEnergy>` that
/// implements `IsotropicTwobodyEnergy`, working around Rust's orphan rules
/// which prevent implementing foreign traits for `Arc<T>` directly.
#[derive(Clone, Debug)]
pub struct ArcPotential(pub Arc<dyn IsotropicTwobodyEnergy>);

impl ArcPotential {
    /// Create a new thread-safe shared potential wrapper.
    pub fn new<T: IsotropicTwobodyEnergy + 'static>(potential: T) -> Self {
        Self(Arc::new(potential))
    }
}

impl Cutoff for ArcPotential {
    fn cutoff(&self) -> f64 {
        self.0.cutoff()
    }
    fn lower_cutoff(&self) -> f64 {
        self.0.lower_cutoff()
    }
}

impl IsotropicTwobodyEnergy for ArcPotential {
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        self.0.isotropic_twobody_energy(distance_squared)
    }

    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        self.0.isotropic_twobody_force(distance_squared)
    }
}

impl Add for Box<dyn IsotropicTwobodyEnergy> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Box::new(Combined::new(self, other))
    }
}

impl Sum for Box<dyn IsotropicTwobodyEnergy> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Box::new(NoInteraction {}), |acc, x| acc + x)
    }
}

/// Plain Coulomb potential combined with Lennard-Jones
pub type CoulombLennardJones = Combined<IonIon<coulomb::pairwise::Plain>, LennardJones>;

/// Yukawa potential combined with Lennard-Jones
pub type YukawaLennardJones = Combined<IonIon<coulomb::pairwise::Yukawa>, LennardJones>;

/// Alias for Weeks-Chandler-Andersen potential
pub type WCA = WeeksChandlerAndersen;

// test Combined
#[test]
fn test_combined() {
    use approx::assert_relative_eq;
    let r2 = 0.5;

    let relative_orientation = RelativeOrientation {
        distance: [f64::sqrt(r2), 0.0, 0.0].into(),
        orientation: [0.0, 1.0, 0.0].into(),
    };

    let pot1 = LennardJones::new(0.5, 1.0);
    let pot2 = Harmonic::new(0.0, 10.0);
    let energy = (
        pot1.isotropic_twobody_energy(r2),
        pot2.isotropic_twobody_energy(r2),
    );
    assert_relative_eq!(energy.0, 112.0);
    assert_relative_eq!(energy.1, 2.5);

    // static dispatch
    let combined = Combined::new(pot1.clone(), pot2.clone());
    assert_relative_eq!(combined.isotropic_twobody_energy(r2), energy.0 + energy.1);
    assert_relative_eq!(
        combined.anisotropic_twobody_energy(&relative_orientation),
        energy.0 + energy.1,
        epsilon = 1e-7
    );

    // dynamic dispatch
    let box1 = Box::new(pot1) as Box<dyn IsotropicTwobodyEnergy>;
    let box2 = Box::new(pot2) as Box<dyn IsotropicTwobodyEnergy>;
    let combined = box1 + box2;

    assert_relative_eq!(combined.isotropic_twobody_energy(r2), energy.0 + energy.1);
    assert_relative_eq!(
        combined.anisotropic_twobody_energy(&relative_orientation),
        energy.0 + energy.1,
        epsilon = 1e-7
    );

    // three combined interactions
    let pot3 = Harmonic::new(1.0, 10.0);
    let energy3 = pot3.isotropic_twobody_energy(r2);
    let box3 = Box::new(pot3) as Box<dyn IsotropicTwobodyEnergy>;
    let combined2 = combined + box3;

    assert_relative_eq!(
        combined2.isotropic_twobody_energy(r2),
        energy.0 + energy.1 + energy3
    );
    assert_relative_eq!(
        combined2.anisotropic_twobody_energy(&relative_orientation),
        energy.0 + energy.1 + energy3,
        epsilon = 1e-7
    );
}

#[test]
fn test_combined_force() {
    use approx::assert_relative_eq;
    let lj = LennardJones::new(1.5, 2.0);
    let h = Harmonic::new(0.0, 10.0);
    let combined = Combined::new(lj.clone(), h.clone());
    let r2 = 6.25;
    assert_relative_eq!(
        combined.isotropic_twobody_force(r2),
        lj.isotropic_twobody_force(r2) + h.isotropic_twobody_force(r2)
    );
}

#[test]
fn test_box_dyn_force() {
    use approx::assert_relative_eq;
    let lj = LennardJones::new(1.5, 2.0);
    let r2 = 6.25;
    let expected = lj.isotropic_twobody_force(r2);
    let boxed: Box<dyn IsotropicTwobodyEnergy> = Box::new(lj);
    assert_relative_eq!(boxed.isotropic_twobody_force(r2), expected);
}

#[test]
fn test_arc_potential_force() {
    use approx::assert_relative_eq;
    let lj = LennardJones::new(1.5, 2.0);
    let r2 = 6.25;
    let expected = lj.isotropic_twobody_force(r2);
    let arc = ArcPotential::new(lj);
    assert_relative_eq!(arc.isotropic_twobody_force(r2), expected);
}

#[test]
fn test_arc_potential() {
    use approx::assert_relative_eq;

    let lj = LennardJones::new(1.0, 1.0);
    let r2 = 1.5_f64.powi(2);
    let expected_energy = lj.isotropic_twobody_energy(r2);
    let expected_cutoff = lj.cutoff();
    let expected_lower_cutoff = lj.lower_cutoff();

    // Create ArcPotential and verify it delegates correctly
    let arc_pot = ArcPotential::new(lj);
    assert_relative_eq!(arc_pot.isotropic_twobody_energy(r2), expected_energy);
    assert_eq!(arc_pot.cutoff(), expected_cutoff);
    assert_relative_eq!(arc_pot.lower_cutoff(), expected_lower_cutoff);

    // Verify Clone works (cheap Arc clone)
    let cloned = arc_pot.clone();
    assert_relative_eq!(cloned.isotropic_twobody_energy(r2), expected_energy);

    // Verify it works with anisotropic interface
    let orientation = RelativeOrientation {
        distance: [1.5, 0.0, 0.0].into(),
        orientation: [0.0, 1.0, 0.0].into(),
    };
    assert_relative_eq!(
        arc_pot.anisotropic_twobody_energy(&orientation),
        expected_energy,
        epsilon = 1e-10
    );
}

#[cfg(test)]
mod range_tests {
    use super::*;

    fn ah(cutoff: f64) -> AshbaughHatch {
        AshbaughHatch::new(LennardJones::new(0.5, 4.0), 0.5, cutoff)
    }

    /// [I2] A sum interacts as far as its longest-ranged term.
    #[test]
    fn combined_range_is_the_longest_term() {
        assert_eq!(Combined::new(ah(20.0), ah(40.0)).cutoff(), 40.0);
    }

    /// [I2] `NoInteraction` is the fold's identity, so a boxed sum keeps the longest range.
    #[test]
    fn boxed_sum_range_ignores_the_empty_seed() {
        let pots: Vec<Box<dyn IsotropicTwobodyEnergy>> =
            vec![Box::new(ah(20.0)), Box::new(ah(40.0))];
        let sum: Box<dyn IsotropicTwobodyEnergy> = pots.into_iter().sum();
        assert_eq!(sum.cutoff(), 40.0);
        assert_eq!(NoInteraction.cutoff(), 0.0);
    }

    /// [I2] One unbounded term makes the whole sum unbounded.
    #[test]
    fn unbounded_term_makes_sum_unbounded() {
        let lj = LennardJones::new(0.5, 4.0);
        assert_eq!(Combined::new(lj, ah(20.0)).cutoff(), f64::INFINITY);
    }

    /// [I1] Every finite-range potential is exactly zero at and beyond its cutoff.
    #[test]
    fn energy_vanishes_beyond_cutoff() {
        use coulomb::pairwise::{EwaldTruncated, Plain, ReactionField};
        use coulomb::permittivity::ConstantPermittivity;
        let water = ConstantPermittivity::new(80.0);
        let pots: Vec<Box<dyn IsotropicTwobodyEnergy>> = vec![
            Box::new(ah(20.0)),
            Box::new(KimHummer::new(-0.5, 6.0).with_cutoff(15.0).unwrap()),
            Box::new(KimHummer::new(0.5, 6.0).with_cutoff(15.0).unwrap()),
            Box::new(WeeksChandlerAndersen::new(0.5, 4.0)),
            Box::new(HardSphere::new(4.0)),
            Box::new(Combined::new(ah(20.0), ah(40.0))),
            Box::new(NoInteraction),
            Box::new(IonIon::new(1.0, water, Plain::new(30.0, None))),
            Box::new(IonIon::new(1.0, water, Plain::new(30.0, Some(10.0)))),
            Box::new(IonIon::new(1.0, water, EwaldTruncated::new(30.0, 0.1))),
            Box::new(IonIon::new(
                1.0,
                water,
                ReactionField::new_shifted(30.0, 80.0, 1.0),
            )),
        ];
        for pot in &pots {
            let rc = pot.cutoff();
            assert!(rc.is_finite(), "{pot:?}");
            for r in [rc, 1.5 * rc] {
                assert_eq!(
                    pot.isotropic_twobody_energy(r * r),
                    0.0,
                    "{pot:?} at r = {r}"
                );
            }
        }
    }

    /// An unbounded potential can't be tabulated from its own range.
    #[test]
    fn splining_an_unbounded_range_is_an_error() {
        let config = SplineConfig::default();
        assert_eq!(
            SplinedPotential::new(&LennardJones::new(0.5, 4.0), config.clone()).unwrap_err(),
            SplineError::Unbounded
        );
    }

    /// NaN fails every comparison, so the config is validated before use.
    #[test]
    fn nan_spline_parameters_are_errors() {
        let lj = LennardJones::new(0.5, 4.0);
        let base = SplineConfig::default;
        let cap = |energy_cap| GridTrim::RepulsiveDecay { energy_cap };
        for config in [
            base().with_rsq_max(f64::NAN),
            base().with_rsq_min(f64::NAN),
            base().with_grid_type(GridType::PowerLaw(f64::NAN)),
            base().with_grid_trim(cap(f64::NAN)),
            base().with_grid_trim(cap(-1.0)),
        ] {
            assert!(SplinedPotential::with_cutoff(&lj, 10.0, config).is_err());
        }
    }
}
