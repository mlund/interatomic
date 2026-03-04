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

//! Runtime custom pair potentials via mathematical expression parsing.
//!
//! Uses the [`exmex`](https://docs.rs/exmex) crate to parse user-provided
//! mathematical expressions and compute symbolic derivatives for forces.
//!
//! # Example
//!
//! ```
//! use interatomic::twobody::{CustomPotential, IsotropicTwobodyEnergy};
//! use interatomic::Cutoff;
//!
//! // Lennard-Jones via expression
//! let lj = CustomPotential::new(
//!     "4*eps*((sigma/r)^12 - (sigma/r)^6)",
//!     &[("eps", 1.0), ("sigma", 1.0)],
//!     2.5,
//! ).unwrap();
//!
//! let r = 1.5;
//! let energy = lj.isotropic_twobody_energy(r * r);
//! ```

use super::IsotropicTwobodyEnergy;
use crate::Cutoff;
use exmex::{Differentiate, Express, FlatEx};
use std::fmt;

/// Errors that can occur when creating a [`CustomPotential`].
#[derive(Debug, Clone)]
pub enum CustomPotentialError {
    /// The expression could not be parsed.
    ParseError(String),
    /// Symbolic differentiation failed.
    DerivativeError(String),
    /// The expression contains unresolved variables (not `r`).
    UnresolvedVariables(Vec<String>),
    /// The cutoff is not positive and finite.
    InvalidCutoff(f64),
}

impl fmt::Display for CustomPotentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "expression parse error: {msg}"),
            Self::DerivativeError(msg) => write!(f, "derivative error: {msg}"),
            Self::UnresolvedVariables(vars) => {
                write!(f, "unresolved variables: {}", vars.join(", "))
            }
            Self::InvalidCutoff(c) => write!(f, "cutoff must be positive and finite, got {c}"),
        }
    }
}

impl std::error::Error for CustomPotentialError {}

/// A pair potential defined by a mathematical expression in `r`.
///
/// The expression is parsed once at construction time, and its symbolic
/// derivative is computed automatically. Both the energy and force are
/// then available for evaluation at any distance.
///
/// # Supported operations
///
/// `+`, `-`, `*`, `/`, `^`, `sin`, `cos`, `tan`, `exp`, `ln`, `log2`,
/// `log10`, `sqrt`, `abs`, and constants `PI`, `E`.
///
/// # Example
///
/// ```
/// use interatomic::twobody::{CustomPotential, IsotropicTwobodyEnergy};
///
/// let harmonic = CustomPotential::new(
///     "0.5 * k * (r - r0)^2",
///     &[("k", 10.0), ("r0", 1.5)],
///     5.0,
/// ).unwrap();
///
/// let r = 2.0;
/// let energy = harmonic.isotropic_twobody_energy(r * r);
/// ```
#[derive(Clone)]
pub struct CustomPotential {
    expression_string: String,
    constants: Vec<(String, f64)>,
    energy_expr: FlatEx<f64>,
    derivative_expr: FlatEx<f64>,
    cutoff: f64,
    lower_cutoff: f64,
}

impl CustomPotential {
    /// Create a new custom potential from a mathematical expression.
    ///
    /// The expression must be a function of a single variable `r` (the distance).
    /// Named parameters are substituted before parsing, so the final expression
    /// must contain only `r` as a variable.
    ///
    /// # Arguments
    ///
    /// * `expression` — mathematical expression string (e.g. `"4*eps*((sigma/r)^12 - (sigma/r)^6)"`)
    /// * `parameters` — named parameter values to substitute into the expression
    /// * `cutoff` — upper cutoff distance (must be positive and finite)
    pub fn new(
        expression: &str,
        parameters: &[(&str, f64)],
        cutoff: f64,
    ) -> Result<Self, CustomPotentialError> {
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(CustomPotentialError::InvalidCutoff(cutoff));
        }

        let substituted = substitute_parameters(expression, parameters);

        let energy_expr: FlatEx<f64> = FlatEx::parse(&substituted)
            .map_err(|e| CustomPotentialError::ParseError(e.to_string()))?;

        // Validate that the only variable is `r`
        let var_names = energy_expr.var_names();
        if var_names.is_empty() {
            return Err(CustomPotentialError::UnresolvedVariables(vec![
                "(expression is constant — no variable 'r' found)".to_string(),
            ]));
        }
        let non_r: Vec<String> = var_names
            .iter()
            .filter(|v| v.as_str() != "r")
            .map(|v| v.to_string())
            .collect();
        if !non_r.is_empty() {
            return Err(CustomPotentialError::UnresolvedVariables(non_r));
        }

        let derivative_expr = energy_expr
            .clone()
            .partial(0)
            .map_err(|e| CustomPotentialError::DerivativeError(e.to_string()))?;

        Ok(Self {
            expression_string: expression.to_string(),
            constants: parameters
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
            energy_expr,
            derivative_expr,
            cutoff,
            lower_cutoff: 0.0,
        })
    }

    /// Set a lower cutoff distance for spline grid bounds.
    pub fn with_lower_cutoff(mut self, lower_cutoff: f64) -> Self {
        self.lower_cutoff = lower_cutoff;
        self
    }

    /// Returns the original expression string.
    pub fn expression(&self) -> &str {
        &self.expression_string
    }

    /// Returns a string representation of the symbolic derivative dU/dr.
    pub fn derivative_expression(&self) -> String {
        self.derivative_expr.to_string()
    }
}

/// Substitute named parameters into an expression string.
///
/// Parameters are sorted by name length (longest first) to avoid
/// substring collisions (e.g. "sigma" before "sig").
fn substitute_parameters(expression: &str, parameters: &[(&str, f64)]) -> String {
    let mut sorted: Vec<(&str, f64)> = parameters.to_vec();
    sorted.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));

    let mut result = expression.to_string();
    for (name, value) in sorted {
        // Wrap in parentheses to handle negative values and avoid operator ambiguity.
        // Use decimal (not scientific) notation since exmex treats `e`/`E` as Euler's number.
        result = result.replace(name, &format!("({value:.17})"));
    }
    result
}

/// Two `CustomPotential`s are equal if they were built from the same expression,
/// constants, and cutoffs. The parsed `FlatEx` fields are derived from these.
impl PartialEq for CustomPotential {
    fn eq(&self, other: &Self) -> bool {
        self.expression_string == other.expression_string
            && self.constants == other.constants
            && self.cutoff == other.cutoff
            && self.lower_cutoff == other.lower_cutoff
    }
}

impl fmt::Debug for CustomPotential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomPotential")
            .field("expression", &self.expression_string)
            .field("cutoff", &self.cutoff)
            .finish()
    }
}

impl Cutoff for CustomPotential {
    fn cutoff(&self) -> f64 {
        self.cutoff
    }

    fn lower_cutoff(&self) -> f64 {
        self.lower_cutoff
    }
}

impl IsotropicTwobodyEnergy for CustomPotential {
    fn isotropic_twobody_energy(&self, distance_squared: f64) -> f64 {
        let r = distance_squared.sqrt();
        self.energy_expr.eval(&[r]).unwrap_or(f64::NAN)
    }

    fn isotropic_twobody_force(&self, distance_squared: f64) -> f64 {
        let r = distance_squared.sqrt();
        let du_dr = self.derivative_expr.eval(&[r]).unwrap_or(f64::NAN);
        // dU/d(r²) = dU/dr · dr/d(r²) = dU/dr · 1/(2r)
        // force = -dU/d(r²) = -dU/dr / (2r)
        -du_dr / (2.0 * r)
    }
}

// Static assertion: CustomPotential must be Send + Sync
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CustomPotential>();
};

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    /// Serialization proxy matching the faunus-rs `CustomPairPotentialBuilder` schema.
    #[derive(Serialize, Deserialize)]
    struct CustomPotentialData {
        function: String,
        cutoff: f64,
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        constants: HashMap<String, f64>,
        #[serde(default, skip_serializing_if = "is_zero")]
        lower_cutoff: f64,
    }

    fn is_zero(v: &f64) -> bool {
        *v == 0.0
    }

    impl Serialize for CustomPotential {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            CustomPotentialData {
                function: self.expression_string.clone(),
                cutoff: self.cutoff,
                constants: self.constants.iter().cloned().collect(),
                lower_cutoff: self.lower_cutoff,
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for CustomPotential {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let data = CustomPotentialData::deserialize(deserializer)?;
            let params: Vec<(&str, f64)> = data
                .constants
                .iter()
                .map(|(k, v)| (k.as_str(), *v))
                .collect();
            CustomPotential::new(&data.function, &params, data.cutoff)
                .map(|p| p.with_lower_cutoff(data.lower_cutoff))
                .map_err(serde::de::Error::custom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twobody::{LennardJones, SplineConfig, SplinedPotential};
    use approx::assert_relative_eq;

    #[test]
    fn lj_expression_vs_native() {
        let eps = 1.0;
        let sigma = 1.0;
        let native = LennardJones::new(eps, sigma);
        let custom = CustomPotential::new(
            "4*eps*((sigma/r)^12 - (sigma/r)^6)",
            &[("eps", eps), ("sigma", sigma)],
            12.0,
        )
        .unwrap();

        for &r in &[0.9, 1.0, 1.1, 1.5, 2.0, 3.0] {
            let r2 = r * r;
            assert_relative_eq!(
                custom.isotropic_twobody_energy(r2),
                native.isotropic_twobody_energy(r2),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn harmonic_energy_and_force() {
        let k = 10.0;
        let r0 = 1.5;
        let custom =
            CustomPotential::new("0.5 * k * (r - r0)^2", &[("k", k), ("r0", r0)], 5.0).unwrap();

        let r = 2.0;
        let r2 = r * r;
        let expected_energy = 0.5 * k * (r - r0).powi(2);
        assert_relative_eq!(
            custom.isotropic_twobody_energy(r2),
            expected_energy,
            epsilon = 1e-10
        );

        // Analytical force: dU/dr = k*(r - r0)
        // isotropic_twobody_force returns -dU/d(r²) = -dU/dr / (2r) = -k*(r - r0) / (2r)
        let expected_force = -k * (r - r0) / (2.0 * r);
        assert_relative_eq!(
            custom.isotropic_twobody_force(r2),
            expected_force,
            epsilon = 1e-10
        );
    }

    #[test]
    fn symbolic_vs_numerical_force() {
        let custom = CustomPotential::new(
            "4*eps*((sigma/r)^12 - (sigma/r)^6)",
            &[("eps", 1.0), ("sigma", 1.0)],
            12.0,
        )
        .unwrap();

        for &r in &[1.0, 1.1, 1.5, 2.0, 3.0] {
            let r2 = r * r;
            let symbolic = custom.isotropic_twobody_force(r2);

            // Central difference: -dU/d(r²) ≈ -(U(r²+h) - U(r²-h)) / (2h)
            let h = 1e-7;
            let numerical = -(custom.isotropic_twobody_energy(r2 + h)
                - custom.isotropic_twobody_energy(r2 - h))
                / (2.0 * h);

            assert_relative_eq!(symbolic, numerical, epsilon = 1e-5);
        }
    }

    #[test]
    fn morse_potential_at_equilibrium() {
        // U(r) = D * (1 - exp(-a*(r - r0)))^2
        // At r = r0: U = 0
        let custom = CustomPotential::new(
            "D * (1 - exp(neg_a * (r - r0)))^2",
            &[("D", 5.0), ("neg_a", -1.5), ("r0", 2.0)],
            10.0,
        )
        .unwrap();

        let r0 = 2.0;
        assert_relative_eq!(
            custom.isotropic_twobody_energy(r0 * r0),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn splined_potential_integration() {
        let native_lj = LennardJones::new(1.0, 1.0);
        let custom = CustomPotential::new(
            "4*eps*((sigma/r)^12 - (sigma/r)^6)",
            &[("eps", 1.0), ("sigma", 1.0)],
            2.5,
        )
        .unwrap()
        .with_lower_cutoff(0.6);

        let splined_native =
            SplinedPotential::with_cutoff(&native_lj, 2.5, SplineConfig::default());
        let splined_custom = SplinedPotential::with_cutoff(&custom, 2.5, SplineConfig::default());

        // Compare splined custom vs splined native (both shifted identically)
        for &r in &[1.0, 1.1, 1.5, 2.0, 2.4] {
            let r2 = r * r;
            assert_relative_eq!(
                splined_custom.isotropic_twobody_energy(r2),
                splined_native.isotropic_twobody_energy(r2),
                epsilon = 1e-2
            );
        }
    }

    #[test]
    fn parse_error() {
        let result = CustomPotential::new("4 * ((( / r", &[], 5.0);
        assert!(matches!(result, Err(CustomPotentialError::ParseError(_))));
    }

    #[test]
    fn unresolved_variables() {
        let result = CustomPotential::new("a * r + b", &[("a", 1.0)], 5.0);
        assert!(matches!(
            result,
            Err(CustomPotentialError::UnresolvedVariables(_))
        ));
    }

    #[test]
    fn constant_expression_error() {
        let result = CustomPotential::new("42", &[], 5.0);
        assert!(matches!(
            result,
            Err(CustomPotentialError::UnresolvedVariables(_))
        ));
    }

    #[test]
    fn invalid_cutoff() {
        let result = CustomPotential::new("r", &[], -1.0);
        assert!(matches!(
            result,
            Err(CustomPotentialError::InvalidCutoff(_))
        ));

        let result = CustomPotential::new("r", &[], f64::INFINITY);
        assert!(matches!(
            result,
            Err(CustomPotentialError::InvalidCutoff(_))
        ));
    }

    #[test]
    fn clone_and_debug() {
        let custom = CustomPotential::new("1/r", &[], 5.0).unwrap();
        let cloned = custom.clone();
        let r2 = 4.0;
        assert_relative_eq!(
            custom.isotropic_twobody_energy(r2),
            cloned.isotropic_twobody_energy(r2)
        );

        let debug = format!("{:?}", custom);
        assert!(debug.contains("CustomPotential"));
        assert!(debug.contains("1/r"));
    }

    #[test]
    fn dynamic_dispatch() {
        let custom = CustomPotential::new("1/r^2", &[], 5.0).unwrap();
        let boxed: Box<dyn IsotropicTwobodyEnergy> = Box::new(custom);
        let r2: f64 = 4.0;
        let r = r2.sqrt();
        assert_relative_eq!(
            boxed.isotropic_twobody_energy(r2),
            1.0 / (r * r),
            epsilon = 1e-10
        );
    }

    #[test]
    fn cutoff_values() {
        let custom = CustomPotential::new("1/r", &[], 5.0)
            .unwrap()
            .with_lower_cutoff(0.5);

        assert_relative_eq!(custom.cutoff(), 5.0);
        assert_relative_eq!(custom.lower_cutoff(), 0.5);
    }

    #[test]
    fn expression_accessors() {
        let custom = CustomPotential::new("1/r^2", &[], 5.0).unwrap();
        assert_eq!(custom.expression(), "1/r^2");
        // The derivative expression should exist and be non-empty
        assert!(!custom.derivative_expression().is_empty());
    }

    #[test]
    fn parameter_substitution_longest_first() {
        // "sigma" should be substituted before "sig" to avoid partial matches
        let custom = CustomPotential::new("(sigma/r)^2", &[("sigma", 2.0)], 5.0).unwrap();

        let r = 1.0;
        assert_relative_eq!(custom.isotropic_twobody_energy(r * r), 4.0, epsilon = 1e-10);
    }

    #[cfg(feature = "serde")]
    mod serde_tests {
        use super::*;

        #[test]
        fn serde_round_trip() {
            let original = CustomPotential::new("1/r^2", &[], 5.0)
                .unwrap()
                .with_lower_cutoff(0.5);

            let json = serde_json::to_string(&original).unwrap();
            let deserialized: CustomPotential = serde_json::from_str(&json).unwrap();

            assert_eq!(deserialized.expression(), original.expression());
            assert_relative_eq!(deserialized.cutoff(), original.cutoff());
            assert_relative_eq!(deserialized.lower_cutoff(), original.lower_cutoff());

            // Verify energy evaluation matches after round-trip
            for &r in &[1.0, 1.5, 2.0, 3.0] {
                let r2 = r * r;
                assert_relative_eq!(
                    deserialized.isotropic_twobody_energy(r2),
                    original.isotropic_twobody_energy(r2),
                    epsilon = 1e-10
                );
            }
        }

        #[test]
        fn serde_with_constants() {
            let original =
                CustomPotential::new("0.5 * k * (r - r0)^2", &[("k", 10.0), ("r0", 1.5)], 5.0)
                    .unwrap();

            let json = serde_json::to_string(&original).unwrap();

            // Verify the serialized form contains the original expression and constants
            let value: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(value["function"], "0.5 * k * (r - r0)^2");
            assert_eq!(value["constants"]["k"], 10.0);
            assert_eq!(value["constants"]["r0"], 1.5);

            let deserialized: CustomPotential = serde_json::from_str(&json).unwrap();

            for &r in &[1.0, 1.5, 2.0, 3.0] {
                let r2 = r * r;
                assert_relative_eq!(
                    deserialized.isotropic_twobody_energy(r2),
                    original.isotropic_twobody_energy(r2),
                    epsilon = 1e-10
                );
            }
        }

        #[test]
        fn deserialize_invalid_expression() {
            let json = r#"{"function": "(((/r", "cutoff": 5.0}"#;
            let result: Result<CustomPotential, _> = serde_json::from_str(json);
            assert!(result.is_err());
        }
    }
}
