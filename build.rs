fn main() {
    println!("cargo::rustc-check-cfg=cfg(dist_build)");
    println!("cargo:rerun-if-env-changed=OXYDLLM_DIST_BUILD");
    println!("cargo:rerun-if-env-changed=OXYDLLM_BUILD_TS_OVERRIDE");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    if std::env::var("OXYDLLM_DIST_BUILD").is_ok() {
        println!("cargo:rustc-cfg=dist_build");
    }

    let ts = if let Ok(v) = std::env::var("OXYDLLM_BUILD_TS_OVERRIDE") {
        v.parse::<u64>().unwrap_or_else(|_| current_ts())
    } else {
        current_ts()
    };
    println!("cargo:rustc-env=OXYDLLM_BUILD_TS={ts}");

    if let Ok(cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        validate_cuda_compute_cap(&cap_str);
        if let Some(cap) = parse_compute_cap(&cap_str) {
            println!("cargo:rustc-env=OXYDLLM_COMPILED_CAP={cap}");
        }
    }
}

// Supported compute capabilities from Ada Lovelace (8.9) onward.
const SUPPORTED_COMPUTE_CAPS: &[(usize, &str)] = &[
    (89, "Ada Lovelace — RTX 40xx, L4, L40/L40S"),
    (90, "Hopper — H100, H200, GH200"),
    (100, "Blackwell — B100, B200, GB200"),
    (103, "Blackwell Ultra — B300, GB300"),
    (110, "Jetson GB — Jetson T4000, T5000"),
    (
        120,
        "Blackwell Desktop — RTX 50xx, RTX PRO 6000/5000/4500/4000/2000",
    ),
    (121, "Blackwell Edge — DGX Spark / NVIDIA GB10"),
];

fn validate_cuda_compute_cap(cap_str: &str) {
    let cap = match parse_compute_cap(cap_str) {
        Some(c) => c,
        None => {
            println!(
                "cargo::error=CUDA_COMPUTE_CAP=\"{cap_str}\" is not a valid compute capability. \
                 Use an integer: e.g. CUDA_COMPUTE_CAP=89 (Ada Lovelace) or CUDA_COMPUTE_CAP=121 (DGX Spark)."
            );
            return;
        }
    };

    if cap < 89 {
        println!(
            "cargo::error=CUDA_COMPUTE_CAP={cap} is below the minimum supported compute capability \
             (8.9 / Ada Lovelace). oxydllm requires >= 8.9. Supported values:\n{}",
            supported_caps_list()
        );
    } else if !SUPPORTED_COMPUTE_CAPS.iter().any(|&(c, _)| c == cap) {
        println!(
            "cargo::error=CUDA_COMPUTE_CAP={cap} is not a supported compute capability. \
             Supported values:\n{}",
            supported_caps_list()
        );
    }
}

fn parse_compute_cap(s: &str) -> Option<usize> {
    // Accept "8.9" → 89, "121" → 121, "12.1" → 121
    if let Some((major, minor)) = s.trim().split_once('.') {
        let maj = major.trim().parse::<usize>().ok()?;
        let min = minor.trim().parse::<usize>().ok()?;
        Some(maj * 10 + min)
    } else {
        s.trim().parse::<usize>().ok()
    }
}

fn supported_caps_list() -> String {
    SUPPORTED_COMPUTE_CAPS
        .iter()
        .map(|&(cap, desc)| format!("  {cap} — {desc}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn current_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
