fn main() {
    println!("cargo::rustc-check-cfg=cfg(dist_build)");
    println!("cargo:rerun-if-env-changed=OXYDLLM_DIST_BUILD");
    println!("cargo:rerun-if-env-changed=OXYDLLM_BUILD_TS_OVERRIDE");

    if std::env::var("OXYDLLM_DIST_BUILD").is_ok() {
        println!("cargo:rustc-cfg=dist_build");
    }

    let ts = if let Ok(v) = std::env::var("OXYDLLM_BUILD_TS_OVERRIDE") {
        v.parse::<u64>().unwrap_or_else(|_| current_ts())
    } else {
        current_ts()
    };
    println!("cargo:rustc-env=OXYDLLM_BUILD_TS={ts}");
}

fn current_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
