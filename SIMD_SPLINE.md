# SIMD and Splined Pair Potentials

This document summarizes findings on SIMD vectorization for tabulated/splined pair potentials,
based on analysis of GROMACS, LAMMPS, and benchmarks on Apple M4.

## Summary

**Scalar splined evaluation is often faster than explicit SIMD** on modern CPUs due to
the gather/scatter bottleneck. Both GROMACS and LAMMPS default to analytical kernels
on wide-SIMD architectures with FMA support.

## Benchmark Results (Apple M4)

Using AshbaughHatch+Yukawa potential (truncated-shifted LJ with hydrophobicity scaling + screened electrostatics).

### Single Evaluation
| Method | Time | Speedup |
|--------|------|---------|
| AH+Yukawa analytical | 2.23 ns | baseline |
| AH+Yukawa splined (scalar) | 1.04 ns | **2.1×** |
| AH+Yukawa splined SIMD (×4) | 5.12 ns (1.28 ns/eval) | 0.8× (slower) |

### Batch Evaluation (10,000 pairs)
| Method | Time | Per-pair | vs Analytical |
|--------|------|----------|---------------|
| AH+Yukawa analytical | 20.4 µs | 2.04 ns | baseline |
| AH+Yukawa splined scalar | **9.7 µs** | 970 ps | **2.1× faster** |
| AH+Yukawa splined SIMD | 18.4 µs | 1.84 ns | slower than scalar |

## Why SIMD Table Lookup is Slow

The spline evaluation loop cannot be efficiently auto-vectorized:

```rust
// Problems for SIMD:
if distance_squared >= self.rsq_max {  // Early exit
    return 0.0;
}
let i = (t as usize).min(self.n - 2);  // Data-dependent index
let c = &self.coeffs[i];               // Scattered memory access
```

**Gather operation costs:**
- AVX2 `vgatherdpd`: ~20 cycles
- AVX-512 `vgatherdpd`: ~8-12 cycles
- ARM NEON: No hardware gather (must emulate with scalar loads)

Each SIMD lane may access a different table index, requiring 4-8 separate memory loads
even with Structure-of-Arrays (SoA) layout.

## AMD Ryzen Considerations

AMD Zen architectures have characteristics that affect SIMD table lookup performance:

**Zen 1-3 (Ryzen 1000-5000 series):**
- 128-bit internal execution units; 256-bit AVX2 ops are split into two µops
- `vgatherdpd` still costs ~20 cycles (similar to Intel)
- The narrower internal width means less penalty for non-vectorized code
- GROMACS notes that Zen's 128-bit units can make tables competitive with analytical

**Zen 4+ (Ryzen 7000+ series):**
- Native 256-bit AVX-512 support with improved gather (~8-12 cycles)
- Better suited for SIMD table lookups than earlier Zen
- Still unlikely to beat scalar spline for small tables that fit in L1/L2

**Expected Ryzen performance:**
| Architecture | SIMD Table Lookup | Recommendation |
|--------------|-------------------|----------------|
| Zen 1-3 | Marginal benefit possible | Test both, likely similar |
| Zen 4+ | May match or beat scalar | Worth enabling SIMD path |

The key insight is that AMD's historically narrower SIMD units made the gather penalty
relatively less severe compared to Intel's wide units. However, scalar spline evaluation
remains competitive because it avoids the gather entirely.

## GROMACS Approach

GROMACS has SIMD table lookup support but **defaults to analytical kernels**:

> "As all modern architectures are wider and support FMA, GROMACS does not use tables
> by default. The only exceptions are kernels without SIMD, which only support tables."
> — [GROMACS 2018 Release Notes](https://manual.gromacs.org/documentation/2018/release-notes/performance.html)

**When GROMACS uses tables:**
- AVX2 with hardware gather instructions
- GPUs (texture units optimized for scattered reads)
- Older 2-wide or 4-wide SIMD without FMA
- AMD Zen (128-bit internal units make tables faster)

**Implementation details:**
- ~10 specialized SIMD routines for table lookups
- `GMX_SIMD_HAVE_GATHER_LOADU_BYSIMDINT_TRANSPOSE` for SIMD offset loads
- Architecture-specific optimizations kept separate from generic SIMD API

References:
- [GROMACS SIMD documentation](https://manual.gromacs.org/current/doxygen/html-full/page_simd.xhtml)
- [GROMACS tabulated interactions](https://manual.gromacs.org/current/reference-manual/special/tabulated-interaction-functions.html)
- [Páll & Hess, "A flexible algorithm for SIMD architectures"](https://doi.org/10.1016/j.cpc.2013.06.003)

## LAMMPS Approach

LAMMPS `pair_style table` offers accelerated versions through packages:
- **GPU**: Uses GPU vector lanes
- **KOKKOS**: Portable parallelism (CPU/GPU)
- **INTEL**: AVX-512 optimized with hardware gather
- **OPENMP**: Thread-level parallelism

The INTEL package specifically exploits AVX-512 gather when available.

Reference: [LAMMPS pair_style table](https://docs.lammps.org/pair_table.html)

## Comparison: Table vs Analytical

| Factor | Table Lookup | Analytical |
|--------|--------------|------------|
| Gather latency | ~20 cycles (AVX2) | 0 |
| FMA utilization | Low (memory bound) | High |
| Cache pressure | High (table in cache) | Low |
| Code complexity | Architecture-specific | Portable |
| Complex potentials | O(1) regardless of complexity | Scales with ops |

## When to Use Splined Potentials

**Good candidates:**
- Complex potentials (AshbaughHatch+Yukawa, many-term expansions)
- Coarse-grained protein models with multiple interaction terms
- User-defined arbitrary potentials
- GPU execution (texture cache efficient for scattered reads)
- Potentials requiring expensive transcendentals (exp, erfc)

**Stick with analytical:**
- Simple potentials (pure LJ, Coulomb)
- Wide SIMD with FMA (AVX2/AVX-512/NEON)
- When memory bandwidth is the bottleneck

## Implementation in This Crate

The `SplinedPotential` provides:
- **AoS layout** (`SplinedPotential`): Best for scalar evaluation
- **SoA layout** (`SplineTableSimd`): Available for future hardware with efficient gather

```rust
// Recommended usage for complex potentials:
let lj = LennardJones::new(epsilon, sigma);
let ah = AshbaughHatch::new(lj, lambda, cutoff);
// Wrap with additional Yukawa term if needed...
let splined = SplinedPotential::new(&ah, SplineConfig::default());

// In inner loop - 2x faster than analytical for complex potentials
let energy = splined.isotropic_twobody_energy(rsq);
```

## Memory Usage

For 20 atom types (210 unique pairs) with default 2000-point tables:
- Per pair: 128 KB
- Total: ~27 MB (fits in L3 cache)

## Future Work

- AVX-512 gather implementation when Rust stabilizes portable SIMD
- GPU kernel export for CUDA/Metal
- Adaptive table resolution based on potential curvature
