use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use interatomic::twobody::{
    AshbaughHatch, IsotropicTwobodyEnergy, KimHummer, LennardJones, Mie, SplineConfig,
    SplinedPotential, WeeksChandlerAndersen,
};
use interatomic::Cutoff;

/// Combined potential: Ashbaugh-Hatch + Yukawa (A/r * exp(-B*r))
///
/// This represents a more realistic and expensive potential than pure LJ,
/// demonstrating the benefits of spline tabulation.
#[derive(Debug, Clone)]
struct AshbaughHatchYukawa {
    /// Ashbaugh-Hatch potential (truncated-shifted LJ with hydrophobicity scaling)
    ashbaugh_hatch: AshbaughHatch,
    /// Yukawa prefactor A (e.g., charge product / dielectric)
    yukawa_a: f64,
    /// Yukawa inverse screening length B (1/Debye length)
    yukawa_b: f64,
}

impl AshbaughHatchYukawa {
    fn new(ashbaugh_hatch: AshbaughHatch, yukawa_a: f64, yukawa_b: f64) -> Self {
        Self {
            ashbaugh_hatch,
            yukawa_a,
            yukawa_b,
        }
    }
}

impl Cutoff for AshbaughHatchYukawa {
    fn cutoff(&self) -> f64 {
        self.ashbaugh_hatch.cutoff()
    }
    fn cutoff_squared(&self) -> f64 {
        self.ashbaugh_hatch.cutoff_squared()
    }
}

// Note: AnisotropicTwobodyEnergy is provided by blanket impl in interatomic

impl IsotropicTwobodyEnergy for AshbaughHatchYukawa {
    fn isotropic_twobody_energy(&self, rsq: f64) -> f64 {
        let r = rsq.sqrt();
        // Ashbaugh-Hatch part (includes cutoff handling)
        let ah = self.ashbaugh_hatch.isotropic_twobody_energy(rsq);
        // Yukawa part: A/r * exp(-B*r)
        let yukawa = self.yukawa_a / r * (-self.yukawa_b * r).exp();
        ah + yukawa
    }

    fn isotropic_twobody_force(&self, rsq: f64) -> f64 {
        let r = rsq.sqrt();
        // Ashbaugh-Hatch force
        let ah_force = self.ashbaugh_hatch.isotropic_twobody_force(rsq);
        // Yukawa force: -d/dr[A/r * exp(-B*r)] = A*exp(-B*r)*(1/r² + B/r)
        let exp_br = (-self.yukawa_b * r).exp();
        let yukawa_force = self.yukawa_a * exp_br * (1.0 / rsq + self.yukawa_b / r);
        ah_force + yukawa_force
    }
}

/// Single-pair benchmarks
fn bench_twobody_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("single");

    let sigma: f64 = 6.0;
    let epsilon: f64 = 0.5;
    let r_squared = (1.2 * sigma).powi(2);

    let lj = LennardJones::new(epsilon, sigma);
    group.bench_function("LennardJones", |b| {
        b.iter(|| lj.isotropic_twobody_energy(black_box(r_squared)))
    });

    let wca = WeeksChandlerAndersen::new(epsilon, sigma);
    group.bench_function("WCA", |b| {
        b.iter(|| wca.isotropic_twobody_energy(black_box(r_squared)))
    });

    let mie: Mie<12, 6> = Mie::new(epsilon, sigma);
    group.bench_function("Mie_12_6", |b| {
        b.iter(|| mie.isotropic_twobody_energy(black_box(r_squared)))
    });

    let kh_attractive = KimHummer::new(-epsilon, sigma);
    group.bench_function("KimHummer_attractive", |b| {
        b.iter(|| kh_attractive.isotropic_twobody_energy(black_box(r_squared)))
    });

    let kh_repulsive = KimHummer::new(epsilon, sigma);
    group.bench_function("KimHummer_repulsive", |b| {
        b.iter(|| kh_repulsive.isotropic_twobody_energy(black_box(r_squared)))
    });

    let kh_neutral = KimHummer::new(0.0, sigma);
    group.bench_function("KimHummer_neutral", |b| {
        b.iter(|| kh_neutral.isotropic_twobody_energy(black_box(r_squared)))
    });

    group.finish();
}

/// Generate test distances (squared) for batch benchmarks
fn generate_distances(n: usize, sigma: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let r = sigma * (0.9 + 1.5 * (i as f64) / (n as f64));
            r * r
        })
        .collect()
}

/// Batch benchmarks to probe vectorization
fn bench_twobody_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch");

    let sigma: f64 = 6.0;
    let epsilon: f64 = 0.5;
    let n_pairs = 10000;
    let distances = generate_distances(n_pairs, sigma);

    // Lennard-Jones
    let lj = LennardJones::new(epsilon, sigma);
    group.bench_with_input(
        BenchmarkId::new("LennardJones", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| {
                dists
                    .iter()
                    .map(|&r2| lj.isotropic_twobody_energy(r2))
                    .sum::<f64>()
            })
        },
    );

    // WCA
    let wca = WeeksChandlerAndersen::new(epsilon, sigma);
    group.bench_with_input(BenchmarkId::new("WCA", n_pairs), &distances, |b, dists| {
        b.iter(|| {
            dists
                .iter()
                .map(|&r2| wca.isotropic_twobody_energy(r2))
                .sum::<f64>()
        })
    });

    // Mie 12-6
    let mie: Mie<12, 6> = Mie::new(epsilon, sigma);
    group.bench_with_input(
        BenchmarkId::new("Mie_12_6", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| {
                dists
                    .iter()
                    .map(|&r2| mie.isotropic_twobody_energy(r2))
                    .sum::<f64>()
            })
        },
    );

    // Kim-Hummer attractive
    let kh_attractive = KimHummer::new(-epsilon, sigma);
    group.bench_with_input(
        BenchmarkId::new("KimHummer_attractive", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| {
                dists
                    .iter()
                    .map(|&r2| kh_attractive.isotropic_twobody_energy(r2))
                    .sum::<f64>()
            })
        },
    );

    // Kim-Hummer repulsive
    let kh_repulsive = KimHummer::new(epsilon, sigma);
    group.bench_with_input(
        BenchmarkId::new("KimHummer_repulsive", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| {
                dists
                    .iter()
                    .map(|&r2| kh_repulsive.isotropic_twobody_energy(r2))
                    .sum::<f64>()
            })
        },
    );

    // Kim-Hummer neutral
    let kh_neutral = KimHummer::new(0.0, sigma);
    group.bench_with_input(
        BenchmarkId::new("KimHummer_neutral", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| {
                dists
                    .iter()
                    .map(|&r2| kh_neutral.isotropic_twobody_energy(r2))
                    .sum::<f64>()
            })
        },
    );

    group.finish();
}

/// Spline single evaluation benchmarks
fn bench_spline_single(c: &mut Criterion) {
    use wide::f64x4;

    let mut group = c.benchmark_group("spline_single");

    let sigma: f64 = 6.0;
    let epsilon: f64 = 0.5;
    let cutoff: f64 = 15.0; // ~2.5σ
    let r_squared = (1.2 * sigma).powi(2);

    // Ashbaugh-Hatch + Yukawa
    let lj = LennardJones::new(epsilon, sigma);
    let ah = AshbaughHatch::new(lj, 0.5, cutoff); // lambda=0.5 for intermediate hydrophobicity
    let ahy = AshbaughHatchYukawa::new(ah, 10.0, 0.5);
    group.bench_function("AH_Yukawa_analytical", |b| {
        b.iter(|| ahy.isotropic_twobody_energy(black_box(r_squared)))
    });

    // Splined AH+Yukawa (scalar)
    let config = SplineConfig {
        n_points: 2000,
        rsq_min: Some((0.8 * sigma).powi(2)),
        ..Default::default()
    };
    let splined_ahy = SplinedPotential::with_cutoff(&ahy, cutoff, config);
    group.bench_function("AH_Yukawa_splined", |b| {
        b.iter(|| splined_ahy.isotropic_twobody_energy(black_box(r_squared)))
    });

    // Splined AH+Yukawa (SIMD x4)
    let simd_table = splined_ahy.to_simd();
    let rsq_x4 = f64x4::splat(r_squared);
    group.bench_function("AH_Yukawa_splined_simd_x4", |b| {
        b.iter(|| simd_table.energy_x4(black_box(rsq_x4)))
    });

    group.finish();
}

/// Spline batch evaluation benchmarks comparing scalar vs SIMD
fn bench_spline_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("spline_batch");
    group.sample_size(100);

    let sigma: f64 = 6.0;
    let epsilon: f64 = 0.5;
    let cutoff: f64 = 15.0;
    let n_pairs = 10000;

    // Generate distances within cutoff range
    let distances: Vec<f64> = (0..n_pairs)
        .map(|i| {
            let r = sigma * (0.9 + 1.5 * (i as f64) / (n_pairs as f64));
            r * r
        })
        .collect();

    // ========== Ashbaugh-Hatch + Yukawa (expensive) ==========

    let lj = LennardJones::new(epsilon, sigma);
    let ah = AshbaughHatch::new(lj, 0.5, cutoff);
    let ahy = AshbaughHatchYukawa::new(ah, 10.0, 0.5);
    group.bench_with_input(
        BenchmarkId::new("AH_Yukawa_analytical", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| {
                dists
                    .iter()
                    .map(|&r2| ahy.isotropic_twobody_energy(r2))
                    .sum::<f64>()
            })
        },
    );

    // ========== Splined AH+Yukawa: scalar ==========

    let config = SplineConfig {
        n_points: 2000,
        rsq_min: Some((0.8 * sigma).powi(2)),
        ..Default::default()
    };
    let splined = SplinedPotential::with_cutoff(&ahy, cutoff, config);

    group.bench_with_input(
        BenchmarkId::new("AH_Yukawa_splined_scalar", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| {
                dists
                    .iter()
                    .map(|&r2| splined.isotropic_twobody_energy(r2))
                    .sum::<f64>()
            })
        },
    );

    // ========== Splined LJ+Yukawa: SIMD SoA ==========

    let simd_table = splined.to_simd();

    group.bench_with_input(
        BenchmarkId::new("AH_Yukawa_splined_simd", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| simd_table.sum_energies_simd(dists))
        },
    );

    // ========== Splined with batch output ==========

    let mut energies = vec![0.0; n_pairs];

    group.bench_with_input(
        BenchmarkId::new("AH_Yukawa_splined_simd_batch", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| {
                simd_table.energies_batch_simd(dists, &mut energies);
                energies.iter().sum::<f64>()
            })
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_twobody_single,
    bench_twobody_batch,
    bench_spline_single,
    bench_spline_batch
);
criterion_main!(benches);
