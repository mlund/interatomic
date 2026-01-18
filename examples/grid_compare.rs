//! Compare grid types for splined potentials.
//!
//! This example compares the accuracy of different grid spacing strategies
//! for cubic Hermite spline interpolation of an AshbaughHatch + Yukawa potential.
//!
//! Run with: `cargo run --example grid_compare`

use coulomb::pairwise::Yukawa;
use coulomb::permittivity::ConstantPermittivity;
use interatomic::twobody::*;
use interatomic::Cutoff;

fn main() {
    // AshbaughHatch parameters
    let epsilon = 0.8; // kJ/mol
    let sigma = 3.0; // Å
    let lambda = 0.5; // hydrophobicity
    let cutoff = 100.0; // Å

    let lj = LennardJones::new(epsilon, sigma);
    let ah = AshbaughHatch::new(lj, lambda, cutoff);

    // Yukawa (screened electrostatics) parameters
    let charge_product = 1.0; // e²
    let permittivity = ConstantPermittivity::new(80.0);
    let debye_length = 50.0; // Å
    let yukawa_scheme = Yukawa::new(cutoff, Some(debye_length));
    let yukawa = IonIon::new(charge_product, permittivity, yukawa_scheme);

    // Combined potential
    let combined = Combined::new(ah, yukawa);
    let rsq_min = combined.lower_cutoff().powi(2);

    // Create splines with different grid types
    let spline_uniform = SplinedPotential::with_cutoff(
        &combined,
        cutoff,
        SplineConfig::default()
            .with_rsq_min(rsq_min)
            .with_grid_type(GridType::UniformRsq),
    );

    let spline_powerlaw = SplinedPotential::with_cutoff(
        &combined,
        cutoff,
        SplineConfig::default()
            .with_rsq_min(rsq_min)
            .with_grid_type(GridType::PowerLaw2),
    );

    let spline_inversersq = SplinedPotential::with_cutoff(
        &combined,
        cutoff,
        SplineConfig::default()
            .with_rsq_min(rsq_min)
            .with_grid_type(GridType::InverseRsq),
    );

    // Print header
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        Grid Type Comparison for AshbaughHatch + Yukawa (n=2000 points)       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  AshbaughHatch: ε={:.1} kJ/mol, σ={:.1} Å, λ={:.1}, cutoff={:.0} Å            ║",
        epsilon, sigma, lambda, cutoff
    );
    println!(
        "║  Yukawa: z₁z₂={:.1}, εᵣ={:.0}, λD={:.0} Å                                     ║",
        charge_product, 80.0, debye_length
    );
    println!(
        "║  r_min = {:.2} Å (from lower_cutoff)                                          ║",
        rsq_min.sqrt()
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Table header
    println!("┌────────┬──────────────┬────────────────────────┬────────────────────────┬────────────────────────┐");
    println!("│  r (Å) │    U_exact   │      UniformRsq        │      PowerLaw2         │      InverseRsq        │");
    println!("│        │   (kJ/mol)   │    abs        rel      │    abs        rel      │    abs        rel      │");
    println!("├────────┼──────────────┼────────────────────────┼────────────────────────┼────────────────────────┤");

    let test_distances = [2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0];

    for r in test_distances {
        let rsq = r * r;
        let u_exact = combined.isotropic_twobody_energy(rsq);

        let u_uniform = spline_uniform.isotropic_twobody_energy(rsq);
        let u_powerlaw = spline_powerlaw.isotropic_twobody_energy(rsq);
        let u_inversersq = spline_inversersq.isotropic_twobody_energy(rsq);

        let abs_u = (u_uniform - u_exact).abs();
        let abs_p = (u_powerlaw - u_exact).abs();
        let abs_i = (u_inversersq - u_exact).abs();

        let rel_u = abs_u / u_exact.abs();
        let rel_p = abs_p / u_exact.abs();
        let rel_i = abs_i / u_exact.abs();

        println!(
            "│ {:>6.1} │ {:>12.4e} │ {:>9.2e}  {:>9.2e} │ {:>9.2e}  {:>9.2e} │ {:>9.2e}  {:>9.2e} │",
            r, u_exact, abs_u, rel_u, abs_p, rel_p, abs_i, rel_i
        );
    }

    println!("└────────┴──────────────┴────────────────────────┴────────────────────────┴────────────────────────┘");
    println!();
    println!(
        "Legend: abs = absolute error, rel = relative error (|U_spline - U_exact| / |U_exact|)"
    );
    println!();
    println!("Summary:");
    println!("  - UniformRsq:  Sparse at short range → large errors where potential is steep");
    println!("  - PowerLaw2:   Balanced sampling → consistent accuracy across all distances");
    println!("  - InverseRsq:  Dense at short range → excellent short-range, poor long-range");
    println!();
}
