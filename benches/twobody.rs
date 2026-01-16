use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use interatomic::twobody::{
    IsotropicTwobodyEnergy, KimHummer, LennardJones, Mie, SplineConfig, SplinedPotential,
    WeeksChandlerAndersen,
};

/// Custom potential: LJ + A/r * exp(-B*r)
/// This is more expensive to compute than pure LJ, showing spline benefits
#[derive(Debug, Clone)]
struct LJYukawa {
    /// LJ epsilon
    epsilon: f64,
    /// LJ sigma
    sigma: f64,
    /// Yukawa prefactor A
    yukawa_a: f64,
    /// Yukawa decay B
    yukawa_b: f64,
}

impl LJYukawa {
    fn new(epsilon: f64, sigma: f64, yukawa_a: f64, yukawa_b: f64) -> Self {
        Self {
            epsilon,
            sigma,
            yukawa_a,
            yukawa_b,
        }
    }
}

// Note: AnisotropicTwobodyEnergy is provided by blanket impl in interatomic

impl IsotropicTwobodyEnergy for LJYukawa {
    fn isotropic_twobody_energy(&self, rsq: f64) -> f64 {
        let r = rsq.sqrt();
        // LJ part
        let s2 = self.sigma * self.sigma / rsq;
        let s6 = s2 * s2 * s2;
        let lj = 4.0 * self.epsilon * s6 * (s6 - 1.0);
        // Yukawa part: A/r * exp(-B*r)
        let yukawa = self.yukawa_a / r * (-self.yukawa_b * r).exp();
        lj + yukawa
    }

    fn isotropic_twobody_force(&self, rsq: f64) -> f64 {
        let r = rsq.sqrt();
        // LJ force: -dU/dr
        let s2 = self.sigma * self.sigma / rsq;
        let s6 = s2 * s2 * s2;
        let lj_force = 24.0 * self.epsilon / r * s6 * (2.0 * s6 - 1.0);
        // Yukawa force: -d/dr[A/r * exp(-B*r)] = A*exp(-B*r)*(1/r² + B/r)
        let exp_br = (-self.yukawa_b * r).exp();
        let yukawa_force = self.yukawa_a * exp_br * (1.0 / rsq + self.yukawa_b / r);
        lj_force + yukawa_force
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
    let mut group = c.benchmark_group("spline_single");

    let sigma: f64 = 6.0;
    let epsilon: f64 = 0.5;
    let cutoff: f64 = 15.0; // ~2.5σ
    let r_squared = (1.2 * sigma).powi(2);

    // Analytical LJ for comparison
    let lj = LennardJones::new(epsilon, sigma);
    group.bench_function("LJ_analytical", |b| {
        b.iter(|| lj.isotropic_twobody_energy(black_box(r_squared)))
    });

    // LJ + Yukawa (more expensive)
    let ljy = LJYukawa::new(epsilon, sigma, 10.0, 0.5);
    group.bench_function("LJYukawa_analytical", |b| {
        b.iter(|| ljy.isotropic_twobody_energy(black_box(r_squared)))
    });

    // Splined LJ
    let config = SplineConfig {
        n_points: 2000,
        rsq_min: Some((0.8 * sigma).powi(2)),
        ..Default::default()
    };
    let splined_lj = SplinedPotential::with_cutoff(&lj, cutoff, config.clone());
    group.bench_function("LJ_splined", |b| {
        b.iter(|| splined_lj.isotropic_twobody_energy(black_box(r_squared)))
    });

    // Splined LJ+Yukawa
    let splined_ljy = SplinedPotential::with_cutoff(&ljy, cutoff, config);
    group.bench_function("LJYukawa_splined", |b| {
        b.iter(|| splined_ljy.isotropic_twobody_energy(black_box(r_squared)))
    });

    group.finish();
}

/// Spline batch evaluation benchmarks comparing scalar vs SIMD
fn bench_spline_batch(c: &mut Criterion) {
    use interatomic::twobody::SplineTableSimd;

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

    // ========== Pure LJ comparison ==========

    let lj = LennardJones::new(epsilon, sigma);
    group.bench_with_input(
        BenchmarkId::new("LJ_analytical", n_pairs),
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

    // ========== LJ + Yukawa (expensive) ==========

    let ljy = LJYukawa::new(epsilon, sigma, 10.0, 0.5);
    group.bench_with_input(
        BenchmarkId::new("LJYukawa_analytical", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| {
                dists
                    .iter()
                    .map(|&r2| ljy.isotropic_twobody_energy(r2))
                    .sum::<f64>()
            })
        },
    );

    // ========== Splined LJ+Yukawa: scalar ==========

    let config = SplineConfig {
        n_points: 2000,
        rsq_min: Some((0.8 * sigma).powi(2)),
        ..Default::default()
    };
    let splined = SplinedPotential::with_cutoff(&ljy, cutoff, config);

    group.bench_with_input(
        BenchmarkId::new("LJYukawa_splined_scalar", n_pairs),
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
        BenchmarkId::new("LJYukawa_splined_simd", n_pairs),
        &distances,
        |b, dists| {
            b.iter(|| simd_table.sum_energies_simd(dists))
        },
    );

    // ========== Splined with batch output ==========

    let mut energies = vec![0.0; n_pairs];

    group.bench_with_input(
        BenchmarkId::new("LJYukawa_splined_simd_batch", n_pairs),
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
