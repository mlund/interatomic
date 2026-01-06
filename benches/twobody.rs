use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use interatomic::twobody::{
    IsotropicTwobodyEnergy, KimHummer, LennardJones, Mie, WeeksChandlerAndersen,
};

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

criterion_group!(benches, bench_twobody_single, bench_twobody_batch);
criterion_main!(benches);
