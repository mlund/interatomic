# Interatomic

[![Crates.io](https://img.shields.io/crates/v/interatomic.svg)](https://crates.io/crates/interatomic)
[![Documentation](https://docs.rs/interatomic/badge.svg)](https://docs.rs/interatomic)
[![License](https://img.shields.io/crates/l/interatomic.svg)](https://github.com/mlund/interatomic)

A Rust library for calculating interatomic interactions such as van der Waals, electrostatics, and other two-body or many-body potentials. Designed for molecular simulations with support for SIMD acceleration and spline tabulation.

## Features

- **Two-body potentials**: Lennard-Jones, Mie, Morse, WCA, FENE, Harmonic, Hard Sphere, Ashbaugh-Hatch, Kim-Hummer, Urey-Bradley
- **Electrostatics**: Ion-ion interactions with Plain Coulomb, Yukawa, and polarization schemes (via [coulomb](https://crates.io/crates/coulomb) crate)
- **Three-body potentials**: Harmonic and Cosine angle/torsion potentials
- **Four-body potentials**: Harmonic and Periodic dihedral potentials
- **Combination rules**: Lorentz-Berthelot, geometric, arithmetic, and Fender-Halsey mixing rules
- **Spline tabulation**: Andrea spline for fast potential evaluation
- **SIMD support**: Vectorized operations via the [wide](https://crates.io/crates/wide) crate
- **Serialization**: Optional serde support for all potentials

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
interatomic = "0.3"
```

With serde support:

```toml
[dependencies]
interatomic = { version = "0.3", features = ["serde"] }
```

## Usage

### Two-body Potentials

```rust
use interatomic::twobody::{LennardJones, IsotropicTwobodyEnergy};

let epsilon = 1.5;  // depth of potential well (kJ/mol)
let sigma = 2.0;    // distance at which potential is zero (Å)
let lj = LennardJones::new(epsilon, sigma);

let r_squared = 4.0;  // squared distance (Å²)
let energy = lj.isotropic_twobody_energy(r_squared);
```

### Combining Potentials

Potentials can be combined using static or dynamic dispatch:

```rust
use interatomic::twobody::{LennardJones, Harmonic, Combined, IsotropicTwobodyEnergy};

// Static dispatch
let lj = LennardJones::new(1.0, 2.0);
let harmonic = Harmonic::new(0.0, 10.0);
let combined = Combined::new(lj, harmonic);

// Dynamic dispatch with Box
let pot1 = Box::new(LennardJones::new(1.0, 2.0)) as Box<dyn IsotropicTwobodyEnergy>;
let pot2 = Box::new(Harmonic::new(0.0, 10.0)) as Box<dyn IsotropicTwobodyEnergy>;
let combined = pot1 + pot2;
```

### Electrostatics

```rust
use interatomic::ELECTRIC_PREFACTOR;

let z1 = 1.0;                    // charge number
let z2 = -1.0;                   // charge number
let r = 7.0;                     // separation (Å)
let rel_dielectric = 80.0;       // relative dielectric constant
let energy = ELECTRIC_PREFACTOR * z1 * z2 / (rel_dielectric * r);  // kJ/mol
```

### Combination Rules

```rust
use interatomic::CombinationRule;

let rule = CombinationRule::LorentzBerthelot;
let (eps1, eps2) = (1.0, 2.0);
let (sig1, sig2) = (3.0, 4.0);
let (epsilon_mixed, sigma_mixed) = rule.mix((eps1, eps2), (sig1, sig2));
```

### Spline Tabulation

For expensive potentials, spline tabulation can significantly improve performance:

```rust
use interatomic::twobody::{LennardJones, SplinedPotential, SplineConfig, IsotropicTwobodyEnergy};

let lj = LennardJones::new(1.0, 3.0);
let config = SplineConfig::default();
let cutoff = 12.0;

let splined = SplinedPotential::with_cutoff(&lj, cutoff, config);
let energy = splined.isotropic_twobody_energy(9.0);
```

## Units

The library uses:
- **Distance**: Ångström (Å)
- **Energy**: kJ/mol

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
