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

use crate::twobody::IsotropicTwobodyEnergy;
use crate::{
    twobody::{LennardJones, WeeksChandlerAndersen},
    Cutoff,
};
use std::fmt;
use std::fmt::{Display, Formatter};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Truncated and shifted Ashbaugh-Hatch
///
/// More information, see <https://doi.org/10.1021/ja802124e>.
///
/// # Examples:
/// ~~~
/// use interatomic::twobody::*;
/// // For λ=1.0 and cutoff = 2^(1/6)σ, we recover WCA:
/// let (epsilon, sigma, lambda) = (1.5, 2.0, 1.0);
/// let cutoff = 2.0_f64.powf(1.0/6.0) * sigma;
/// let lj = LennardJones::new(epsilon, sigma);
/// let ah = AshbaughHatch::new(lj.clone(), lambda, cutoff);
/// let wca = WeeksChandlerAndersen::from(lj.clone());
/// let r2 = sigma * sigma;
/// assert_eq!(ah.isotropic_twobody_energy(r2), wca.isotropic_twobody_energy(r2));
/// assert!(ah.is_wca());
/// assert!(AshbaughHatch::new_wca(epsilon, sigma).is_wca());
/// ~~~
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize),
    serde(into = "AshbaughHatchSerde")
)]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct AshbaughHatch {
    pub(crate) lennard_jones: LennardJones,
    lambda: f64,
    cutoff: f64,
}

/// Serde helper for deserializing with optional `wca` flag.
#[cfg(feature = "serde")]
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AshbaughHatchSerde {
    #[serde(flatten)]
    lennard_jones: LennardJones,
    #[serde(alias = "λ", default)]
    lambda: f64,
    #[serde(alias = "rc", default)]
    cutoff: f64,
    /// If true, set λ=1 and cutoff=σ·2^(1/6) (purely repulsive WCA).
    #[serde(default)]
    wca: bool,
}

#[cfg(feature = "serde")]
impl From<AshbaughHatchSerde> for AshbaughHatch {
    fn from(raw: AshbaughHatchSerde) -> Self {
        if raw.wca {
            Self::new_wca(raw.lennard_jones.epsilon(), raw.lennard_jones.sigma())
        } else {
            Self::new(raw.lennard_jones, raw.lambda, raw.cutoff)
        }
    }
}

#[cfg(feature = "serde")]
impl From<AshbaughHatch> for AshbaughHatchSerde {
    fn from(ah: AshbaughHatch) -> Self {
        Self {
            wca: ah.is_wca(),
            lennard_jones: ah.lennard_jones,
            lambda: ah.lambda,
            cutoff: ah.cutoff,
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for AshbaughHatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        AshbaughHatchSerde::deserialize(deserializer).map(Into::into)
    }
}

impl AshbaughHatch {
    /// Create a new Ashbaugh-Hatch potential.
    pub const fn new(lennard_jones: LennardJones, lambda: f64, cutoff: f64) -> Self {
        Self {
            lennard_jones,
            lambda,
            cutoff,
        }
    }

    /// Create a WCA-equivalent potential (λ=1, cutoff=σ·2^(1/6)).
    pub fn new_wca(epsilon: f64, sigma: f64) -> Self {
        let lj = LennardJones::new(epsilon, sigma);
        let cutoff = sigma * 2.0_f64.powf(1.0 / 6.0);
        Self::new(lj, 1.0, cutoff)
    }

    /// True if equivalent to WCA (λ=1, cutoff ≈ σ·2^(1/6)).
    pub fn is_wca(&self) -> bool {
        self.lambda == 1.0 && {
            let wca_cutoff = self.lennard_jones.sigma() * 2.0_f64.powf(1.0 / 6.0);
            (self.cutoff - wca_cutoff).abs() < 1e-6 * self.lennard_jones.sigma()
        }
    }
}

impl Cutoff for AshbaughHatch {
    fn cutoff_squared(&self) -> f64 {
        self.cutoff * self.cutoff
    }
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
    fn lower_cutoff(&self) -> f64 {
        self.lennard_jones.lower_cutoff()
    }
}

impl IsotropicTwobodyEnergy for AshbaughHatch {
    #[inline(always)]
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        if distance_squared > self.cutoff_squared() {
            return 0.0;
        }

        let lj = self
            .lennard_jones
            .isotropic_twobody_energy(distance_squared);

        let lj_rc = self
            .lennard_jones
            .isotropic_twobody_energy(self.cutoff_squared());

        if distance_squared
            <= self.lennard_jones.sigma_squared * WeeksChandlerAndersen::TWOTOTWOSIXTH
        {
            self.lennard_jones
                .epsilon()
                .mul_add(1.0 - self.lambda, self.lambda.mul_add(-lj_rc, lj))
        } else {
            self.lambda * (lj - lj_rc)
        }
    }

    #[inline(always)]
    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        if distance_squared > self.cutoff_squared() {
            return 0.0;
        }
        self.lambda * self.lennard_jones.isotropic_twobody_force(distance_squared)
    }
}

impl Display for AshbaughHatch {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.is_wca() {
            write!(
                f,
                "Ashbaugh-Hatch (WCA) with ε = {:.3}, σ = {:.3}",
                self.lennard_jones.epsilon(),
                self.lennard_jones.sigma()
            )
        } else {
            write!(
                f,
                "Ashbaugh-Hatch with λ = {:.3}, cutoff = {:.3}, ε = {:.3}, σ = {:.3}",
                self.lambda,
                self.cutoff,
                self.lennard_jones.epsilon(),
                self.lennard_jones.sigma()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ashbaugh_hatch_force() {
        let lj = LennardJones::new(1.5, 2.0);
        let cutoff = 5.0;
        // λ=1: identical to LJ
        let ah1 = AshbaughHatch::new(lj.clone(), 1.0, cutoff);
        assert_relative_eq!(
            ah1.isotropic_twobody_force(6.25),
            lj.isotropic_twobody_force(6.25)
        );
        // λ=0.5: half the LJ force
        let ah05 = AshbaughHatch::new(lj.clone(), 0.5, cutoff);
        assert_relative_eq!(
            ah05.isotropic_twobody_force(6.25),
            0.5 * lj.isotropic_twobody_force(6.25)
        );
        // Beyond cutoff: zero
        assert_relative_eq!(ah05.isotropic_twobody_force(30.0), 0.0);
    }

    #[test]
    fn new_wca_is_wca() {
        let ah = AshbaughHatch::new_wca(1.5, 2.0);
        assert!(ah.is_wca());
        // Energy matches WCA
        let wca = WeeksChandlerAndersen::from(LennardJones::new(1.5, 2.0));
        let r2 = 3.0; // inside core
        assert_relative_eq!(
            ah.isotropic_twobody_energy(r2),
            wca.isotropic_twobody_energy(r2),
            epsilon = 1e-10
        );
        // Zero beyond cutoff
        let r2_far = 10.0;
        assert_relative_eq!(ah.isotropic_twobody_energy(r2_far), 0.0);
    }

    #[test]
    fn normal_ah_is_not_wca() {
        let lj = LennardJones::new(1.5, 2.0);
        assert!(!AshbaughHatch::new(lj.clone(), 0.5, 5.0).is_wca());
        assert!(!AshbaughHatch::new(lj, 1.0, 5.0).is_wca());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_wca_flag() {
        let yaml = r#"{ epsilon: 1.5, sigma: 2.0, wca: true }"#;
        let ah: AshbaughHatch = serde_json::from_str(yaml).unwrap();
        assert!(ah.is_wca());
        assert_eq!(ah.lambda, 1.0);
    }
}
