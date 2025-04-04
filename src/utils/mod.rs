/// Possible architecture-dependent features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdInstructionSet {
    #[cfg(target_arch = "x86_64")]
    AVX512,

    #[cfg(target_arch = "x86_64")]
    AVX2,

    #[cfg(target_arch = "x86_64")]
    AVX,

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE4_2,

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE4_1,

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE2,

    #[cfg(target_arch = "aarch64")]
    Neon,

    #[cfg(any(target_arch = "mips", target_arch = "mips64"))]
    MSA, // MIPS SIMD

    /// Altivec
    #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
    Altivec,
    
    /// Vector Extensions
    #[cfg(target_arch = "powerpc64")]
    Vsx,

    /// RISC-V Vector Extensions
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    RVV,

    /// Unknown or unsupported CPU features
    None,
}

/// Detects the best available SIMD instruction set
/// # Safety
/// actually safe
#[allow(unreachable_code)]
pub unsafe fn detect_simd_instruction_set() -> SimdInstructionSet {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return if is_x86_feature_detected!("avx512f") {
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
    };

    #[cfg(target_arch = "aarch64")]
    {
        return if is_aarch64_feature_detected!("neon") {
            SimdInstructionSet::Neon
        } else {
            SimdInstructionSet::None
        };
    }

    #[cfg(any(target_arch = "mips", target_arch = "mips64"))]
    // MIPS SIMD Architecture (MSA)
    return if cfg!(target_feature = "msa") {
        SimdInstructionSet::MSA
    } else {
        SimdInstructionSet::None
    };

    #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
    // PowerPC Altivec (VMX)
    return if cfg!(target_feature = "altivec") {
        SimdInstructionSet::Altivec
    } else {
        SimdInstructionSet::None
    };

    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    // RISC-V Vector Extension (RVV)
    return if cfg!(target_feature = "rvv") {
        SimdInstructionSet::RVV
    } else {
        SimdInstructionSet::None
    };

    SimdInstructionSet::None
}

// Prints System Information
pub fn print_system_info() {
    unsafe {
        let simd = detect_simd_instruction_set();
        println!("Detected SIMD Instruction Set: {:?}", simd);
    }
}
