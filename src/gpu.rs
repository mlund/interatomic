//! GPU-friendly spline data structures for pair and angular potentials.
//!
//! Provides `GpuSplineData` that packs spline coefficients and parameters
//! into GPU-ready buffers with 32-byte aligned structs suitable for WebGPU/WGSL.
//!
//! # Example
//!
//! ```
//! use interatomic::gpu::{GpuSplineData, SPLINE_WGSL};
//! use interatomic::twobody::{LennardJones, SplinedPotential, SplineConfig, GridType};
//!
//! let lj = LennardJones::new(1.0, 1.0);
//! let splined = SplinedPotential::with_cutoff(
//!     &lj, 2.5,
//!     SplineConfig::default().with_grid_type(GridType::PowerLaw2),
//! );
//!
//! let mut gpu = GpuSplineData::new();
//! gpu.push(&splined);
//!
//! let params_bytes = gpu.params_as_bytes();
//! let coeffs_bytes = gpu.coefficients_as_bytes();
//! assert!(!params_bytes.is_empty());
//! assert!(!coeffs_bytes.is_empty());
//! ```

use crate::twobody::{GridType, SplineCoeffs, SplinedPotential};
use bytemuck::{Pod, Zeroable};

/// WGSL shader snippet for spline evaluation on GPU.
pub const SPLINE_WGSL: &str = include_str!("shaders/spline.wgsl");

/// GPU-ready spline parameters for a single pair potential (32 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct GpuSplineParams {
    /// Minimum distance of the spline grid.
    pub r_min: f32,
    /// Maximum distance (cutoff) of the spline grid.
    pub r_max: f32,
    /// Grid type: 0 = PowerLaw2, 1 = InverseRsq.
    pub grid_type: u32,
    /// Number of spline coefficient entries.
    pub n_coeffs: u32,
    /// Offset into the shared coefficient array.
    pub coeff_offset: u32,
    /// Inverse grid spacing.
    pub inv_delta: f32,
    /// `1 / r_max²`, used by InverseRsq grid.
    pub w_min: f32,
    /// Force at `r_min` for linear extrapolation below the grid.
    pub f_at_rmin: f32,
}

// SAFETY: GpuSplineParams is #[repr(C)] with only f32/u32 fields, no padding holes.
unsafe impl Zeroable for GpuSplineParams {}
unsafe impl Pod for GpuSplineParams {}

/// GPU-ready spline coefficients for one grid interval (32 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct GpuSplineCoeffs {
    /// Energy polynomial coefficients `[A₀, A₁, A₂, A₃]`
    pub u: [f32; 4],
    /// Force polynomial coefficients `[B₀, B₁, B₂, B₃]`
    pub f: [f32; 4],
}

// SAFETY: GpuSplineCoeffs is #[repr(C)] with only f32 fields.
unsafe impl Zeroable for GpuSplineCoeffs {}
unsafe impl Pod for GpuSplineCoeffs {}

/// GPU-ready parameters for an angular spline (32 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct GpuAngleSplineParams {
    /// Minimum angle of the spline grid (radians).
    pub angle_min: f32,
    /// Maximum angle of the spline grid (radians).
    pub angle_max: f32,
    /// Number of spline coefficient entries.
    pub n_coeffs: u32,
    /// Offset into the shared coefficient array.
    pub coeff_offset: u32,
    /// Inverse grid spacing.
    pub inv_delta: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// SAFETY: GpuAngleSplineParams is #[repr(C)] with only f32/u32 fields.
unsafe impl Zeroable for GpuAngleSplineParams {}
unsafe impl Pod for GpuAngleSplineParams {}

/// Convert f64 spline coefficients to GPU f32 format and append to a buffer.
fn pack_coeffs(dst: &mut Vec<GpuSplineCoeffs>, src: &[SplineCoeffs]) {
    dst.extend(src.iter().map(|c| GpuSplineCoeffs {
        u: [c.u[0] as f32, c.u[1] as f32, c.u[2] as f32, c.u[3] as f32],
        f: [c.f[0] as f32, c.f[1] as f32, c.f[2] as f32, c.f[3] as f32],
    }));
}

/// Aggregated GPU spline data for multiple potentials.
///
/// Coefficients from all potentials are stored contiguously. Each
/// potential's `coeff_offset` indexes into the shared coefficient array.
#[derive(Clone, Debug, Default)]
pub struct GpuSplineData {
    /// Shared coefficient storage for all potentials (pair + angular).
    pub coefficients: Vec<GpuSplineCoeffs>,
    /// Per-potential pair spline parameters.
    pub params: Vec<GpuSplineParams>,
    /// Per-potential angular spline parameters.
    pub angle_params: Vec<GpuAngleSplineParams>,
}

impl GpuSplineData {
    /// Create empty GPU spline data.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a splined pair potential to the GPU data.
    ///
    /// Only `PowerLaw2` and `InverseRsq` grids are supported for GPU evaluation.
    ///
    /// # Panics
    /// Panics if the grid type is not `PowerLaw2` or `InverseRsq`.
    pub fn push(&mut self, spline: &SplinedPotential) {
        let grid_type_u32 = match spline.grid_type() {
            GridType::PowerLaw2 => 0u32,
            GridType::InverseRsq => 1u32,
            other => panic!(
                "GPU spline only supports PowerLaw2 and InverseRsq grids, got {:?}",
                other
            ),
        };

        let coeff_offset = self.coefficients.len() as u32;
        let n_coeffs = spline.coefficients().len() as u32;
        pack_coeffs(&mut self.coefficients, spline.coefficients());

        let r_max = spline.r_max();
        self.params.push(GpuSplineParams {
            r_min: spline.r_min() as f32,
            r_max: r_max as f32,
            grid_type: grid_type_u32,
            n_coeffs,
            coeff_offset,
            inv_delta: spline.inv_delta() as f32,
            w_min: (1.0 / (r_max * r_max)) as f32,
            f_at_rmin: spline.f_at_rmin() as f32,
        });
    }

    /// Add a splined bond angle potential to the GPU data.
    pub fn push_angle(&mut self, spline: &crate::threebody::SplinedAngle) {
        self.push_angle_impl(
            spline.coefficients(),
            spline.angle_min(),
            spline.angle_max(),
            spline.inv_delta(),
        );
    }

    /// Add a splined dihedral potential to the GPU data.
    pub fn push_dihedral(&mut self, spline: &crate::fourbody::SplinedDihedral) {
        self.push_angle_impl(
            spline.coefficients(),
            spline.angle_min(),
            spline.angle_max(),
            spline.inv_delta(),
        );
    }

    /// Shared implementation for angular spline packing.
    fn push_angle_impl(
        &mut self,
        coefficients: &[SplineCoeffs],
        angle_min: f64,
        angle_max: f64,
        inv_delta: f64,
    ) {
        let coeff_offset = self.coefficients.len() as u32;
        let n_coeffs = coefficients.len() as u32;
        pack_coeffs(&mut self.coefficients, coefficients);

        self.angle_params.push(GpuAngleSplineParams {
            angle_min: angle_min as f32,
            angle_max: angle_max as f32,
            n_coeffs,
            coeff_offset,
            inv_delta: inv_delta as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        });
    }

    /// Create GPU spline data from an iterator of splined potentials.
    pub fn from_potentials<'a>(iter: impl IntoIterator<Item = &'a SplinedPotential>) -> Self {
        let mut data = Self::new();
        for spline in iter {
            data.push(spline);
        }
        data
    }

    /// Get pair spline parameters as a byte slice for GPU upload.
    pub fn params_as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.params)
    }

    /// Get all spline coefficients as a byte slice for GPU upload.
    pub fn coefficients_as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.coefficients)
    }

    /// Get angular spline parameters as a byte slice for GPU upload.
    pub fn angle_params_as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.angle_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twobody::{LennardJones, SplineConfig};

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<GpuSplineParams>(), 32);
        assert_eq!(std::mem::size_of::<GpuSplineCoeffs>(), 32);
        assert_eq!(std::mem::size_of::<GpuAngleSplineParams>(), 32);
    }

    #[test]
    fn test_powerlaw2_conversion() {
        let lj = LennardJones::new(1.0, 1.0);
        let splined = SplinedPotential::with_cutoff(
            &lj,
            2.5,
            SplineConfig::default().with_grid_type(GridType::PowerLaw2),
        );
        let mut gpu = GpuSplineData::new();
        gpu.push(&splined);

        assert_eq!(gpu.params.len(), 1);
        assert_eq!(gpu.params[0].grid_type, 0);
        assert_eq!(gpu.params[0].coeff_offset, 0);
        assert_eq!(gpu.params[0].n_coeffs, gpu.coefficients.len() as u32);
    }

    #[test]
    fn test_inverse_rsq_conversion() {
        let lj = LennardJones::new(1.0, 1.0);
        let splined = SplinedPotential::with_cutoff(
            &lj,
            2.5,
            SplineConfig::default().with_grid_type(GridType::InverseRsq),
        );
        let mut gpu = GpuSplineData::new();
        gpu.push(&splined);

        assert_eq!(gpu.params[0].grid_type, 1);
        let expected_w_min = 1.0f32 / (2.5f32 * 2.5f32);
        assert!((gpu.params[0].w_min - expected_w_min).abs() < 1e-6);
    }

    #[test]
    fn test_multi_potential_offsets() {
        let lj1 = LennardJones::new(1.0, 1.0);
        let lj2 = LennardJones::new(2.0, 0.5);
        let config = SplineConfig {
            n_points: 100,
            ..SplineConfig::default()
        };
        let s1 = SplinedPotential::with_cutoff(&lj1, 2.5, config.clone());
        let s2 = SplinedPotential::with_cutoff(&lj2, 3.0, config);

        let mut gpu = GpuSplineData::new();
        gpu.push(&s1);
        let n1 = gpu.coefficients.len() as u32;
        gpu.push(&s2);

        assert_eq!(gpu.params[0].coeff_offset, 0);
        assert_eq!(gpu.params[1].coeff_offset, n1);
        assert_eq!(gpu.params.len(), 2);
    }

    #[test]
    fn test_byte_roundtrip() {
        let lj = LennardJones::new(1.0, 1.0);
        let splined = SplinedPotential::with_cutoff(
            &lj,
            2.5,
            SplineConfig::default().with_grid_type(GridType::PowerLaw2),
        );
        let mut gpu = GpuSplineData::new();
        gpu.push(&splined);

        let params_bytes = gpu.params_as_bytes();
        let coeffs_bytes = gpu.coefficients_as_bytes();

        assert_eq!(params_bytes.len(), 32);
        assert_eq!(coeffs_bytes.len(), gpu.coefficients.len() * 32);

        // Roundtrip params
        let params_back: &[GpuSplineParams] = bytemuck::cast_slice(params_bytes);
        assert_eq!(params_back[0], gpu.params[0]);

        // Roundtrip coefficients
        let coeffs_back: &[GpuSplineCoeffs] = bytemuck::cast_slice(coeffs_bytes);
        assert_eq!(coeffs_back.len(), gpu.coefficients.len());
        assert_eq!(coeffs_back[0], gpu.coefficients[0]);
    }

    #[test]
    #[should_panic(expected = "GPU spline only supports")]
    fn test_reject_unsupported_grid() {
        let lj = LennardJones::new(1.0, 1.0);
        let splined = SplinedPotential::with_cutoff(
            &lj,
            2.5,
            SplineConfig::default().with_grid_type(GridType::UniformR),
        );
        let mut gpu = GpuSplineData::new();
        gpu.push(&splined);
    }

    #[test]
    fn test_from_potentials() {
        let lj = LennardJones::new(1.0, 1.0);
        let config = SplineConfig::default();
        let s1 = SplinedPotential::with_cutoff(&lj, 2.5, config.clone());
        let s2 = SplinedPotential::with_cutoff(&lj, 3.0, config);
        let potentials = vec![s1, s2];
        let gpu = GpuSplineData::from_potentials(&potentials);
        assert_eq!(gpu.params.len(), 2);
    }

    #[test]
    fn test_angle_spline_gpu() {
        use crate::threebody::{HarmonicTorsion, SplinedAngle};
        let pot = HarmonicTorsion::new(std::f64::consts::FRAC_PI_2, 100.0);
        let splined = SplinedAngle::new(&pot, 200);

        let mut gpu = GpuSplineData::new();
        gpu.push_angle(&splined);

        assert_eq!(gpu.angle_params.len(), 1);
        assert!((gpu.angle_params[0].angle_min - 0.0).abs() < 1e-6);
        assert!((gpu.angle_params[0].angle_max - std::f32::consts::PI).abs() < 1e-5);
        assert!(!gpu.coefficients.is_empty());
    }

    #[test]
    fn test_dihedral_spline_gpu() {
        use crate::fourbody::{HarmonicDihedral, SplinedDihedral};
        let pot = HarmonicDihedral::new(0.0, 50.0);
        let splined = SplinedDihedral::new(&pot, 200);

        let mut gpu = GpuSplineData::new();
        gpu.push_dihedral(&splined);

        assert_eq!(gpu.angle_params.len(), 1);
        assert!((gpu.angle_params[0].angle_min - (-std::f32::consts::PI)).abs() < 1e-5);
        assert!((gpu.angle_params[0].angle_max - std::f32::consts::PI).abs() < 1e-5);
    }
}
