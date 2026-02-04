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

//! Implementation of the Morse potential (work in progress).

#![allow(dead_code)]

use super::IsotropicTwobodyEnergy;
use crate::Cutoff;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Morse potential.
/// See <https://en.wikipedia.org/wiki/Morse_potential>.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(deny_unknown_fields)
)]
pub struct Morse {
    #[cfg_attr(feature = "serde", serde(rename = "req"))]
    equilibrium_distance: f64,
    #[cfg_attr(feature = "serde", serde(rename = "d"))]
    well_depth: f64,
    #[cfg_attr(feature = "serde", serde(rename = "k"))]
    force_constant: f64,
    // shouldn't the parameter a, the 'width of the well' also be present?
}

impl Morse {
    /// Create a new Morse potential with equilibrium distance, well depth, and force constant.
    pub const fn new(equilibrium_distance: f64, well_depth: f64, force_constant: f64) -> Self {
        Self {
            equilibrium_distance,
            well_depth,
            force_constant,
        }
    }
}

impl IsotropicTwobodyEnergy for Morse {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, _distance_squared: f64) -> f64 {
        todo!("Morse potential is not yet implemented");
    }
}

impl Cutoff for Morse {
    fn cutoff(&self) -> f64 {
        f64::INFINITY
    }
}
