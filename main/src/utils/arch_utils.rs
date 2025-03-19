#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArchitecture {
    X86_64,
    AArch64,
    Mips,
    PowerPC,
    RiscV,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdInstructionSet {
    AVX512,
    AVX2,
    AVX,
    SSE4_2,
    SSE4_1,
    SSE2,
    Neon,
    MSA,      // MIPS SIMD
    Altivec,  // PowerPC SIMD
    RVV,      // RISC-V Vector Extension
    None,
}

// Detects the CPU architecture at runtime
pub fn detect_cpu_architecture() -> CpuArchitecture {
    if cfg!(target_arch = "x86_64") {
        CpuArchitecture::X86_64
    } else if cfg!(target_arch = "aarch64") {
        CpuArchitecture::AArch64
    } else if cfg!(target_arch = "mips") || cfg!(target_arch = "mips64") {
        CpuArchitecture::Mips
    } else if cfg!(target_arch = "powerpc") || cfg!(target_arch = "powerpc64") {
        CpuArchitecture::PowerPC
    } else if cfg!(target_arch = "riscv32") || cfg!(target_arch = "riscv64") {
        CpuArchitecture::RiscV
    } else {
        CpuArchitecture::Unknown
    }
}

// Detects the best available SIMD instruction set
pub fn detect_simd_instruction_set() -> SimdInstructionSet {
    match detect_cpu_architecture() {
        CpuArchitecture::X86_64 => {
            unsafe {
                if is_x86_feature_detected!("avx512f") {
                    SimdInstructionSet::AVX512
                } else if is_x86_feature_detected!("avx2") {
                    SimdInstructionSet::AVX2
                } else if is_x86_feature_detected!("avx") {
                    SimdInstructionSet::AVX
                } else if is_x86_feature_detected!("sse4.2") {
                    SimdInstructionSet::SSE4_2
                } else if is_x86_feature_detected!("sse4.1") {
                    SimdInstructionSet::SSE4_1
                } else if is_x86_feature_detected!("sse2") {
                    SimdInstructionSet::SSE2
                } else {
                    SimdInstructionSet::None
                }
            }
        }
        CpuArchitecture::AArch64 => {
            #[cfg(target_arch = "aarch64")]
            {
                if is_aarch64_feature_detected!("neon") {
                    SimdInstructionSet::Neon
                } else {
                    SimdInstructionSet::None
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                SimdInstructionSet::None
            }
        }
        CpuArchitecture::Mips => {
            // MIPS SIMD Architecture (MSA)
            if cfg!(target_feature = "msa") {
                SimdInstructionSet::MSA
            } else {
                SimdInstructionSet::None
            }
        }
        CpuArchitecture::PowerPC => {
            // PowerPC Altivec (VMX)
            if cfg!(target_feature = "altivec") {
                SimdInstructionSet::Altivec
            } else {
                SimdInstructionSet::None
            }
        }
        CpuArchitecture::RiscV => {
            // RISC-V Vector Extension (RVV)
            if cfg!(target_feature = "rvv") {
                SimdInstructionSet::RVV
            } else {
                SimdInstructionSet::None
            }
        }
        CpuArchitecture::Unknown => SimdInstructionSet::None,
    }
}

// Prints System Information
pub fn print_system_info() {
    let arch = detect_cpu_architecture();
    let simd = detect_simd_instruction_set();

    println!("Detected CPU Architecture: {:?}", arch);
    println!("Detected SIMD Instruction Set: {:?}", simd);
}


