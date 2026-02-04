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

//! Implementation of the enum for holding twobody potentials.

/// Enum representing different twobody potential types.
pub enum TwobodyPotential {
    /// Ashbaugh-Hatch potential (truncated and shifted LJ with hydrophobicity scaling)
    AshbaughHatch(super::AshbaughHatch),
    /// Combined potential (sum of two potentials)
    Combined,
    /// Finitely extensible nonlinear elastic potential
    FENE(super::FENE),
    /// Hard sphere potential
    HardSphere(super::HardSphere),
    /// Harmonic potential
    Harmonic(super::Harmonic),
    /// Kim-Hummer potential
    KimHummer(super::KimHummer),
    /// Lennard-Jones potential
    LennardJones(super::LennardJones),
    /// Weeks-Chandler-Andersen potential (repulsive LJ)
    WeeksChandlerAndersen(super::WeeksChandlerAndersen),
}
