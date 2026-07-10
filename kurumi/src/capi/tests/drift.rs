//! ABI drift guards: scrape the `ku_*` exports from the C-ABI source and reconcile them
//! against kurumi.h (header drift) and the kurumi_core op builders (missing-wrapper drift).

// every file that defines exported symbols (the ops families, the handle/eval/tensor
// surfaces, and the parent module for ku_last_error).
const CAPI_SRCS: &[&str] = &[
    include_str!("../ops.rs"),
    include_str!("../ops/contract.rs"),
    include_str!("../ops/distance.rs"),
    include_str!("../ops/indexing.rs"),
    include_str!("../ops/linalg.rs"),
    include_str!("../ops/movement.rs"),
    include_str!("../ops/nn.rs"),
    include_str!("../ops/nn/activation.rs"),
    include_str!("../ops/nn/attention.rs"),
    include_str!("../ops/nn/conv.rs"),
    include_str!("../ops/nn/loss.rs"),
    include_str!("../ops/nn/norm.rs"),
    include_str!("../ops/nn/pool.rs"),
    include_str!("../ops/random.rs"),
    include_str!("../ops/signal.rs"),
    include_str!("../ops/spatial.rs"),
    include_str!("../ops/stats.rs"),
    include_str!("../graph.rs"),
    include_str!("../graph/leaves.rs"),
    include_str!("../graph/serialize.rs"),
    include_str!("../eval.rs"),
    include_str!("../tensor.rs"),
    include_str!("../../capi.rs"),
];

fn ident_prefix(s: &str) -> String {
    s.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect()
}
fn trailing_ident(s: &str) -> String {
    let rev: String = s.trim_end().chars().rev().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    rev.chars().rev().collect()
}

// The exported `ku_*` symbols, scraped from the ABI source: explicit `extern "C" fn ku_*`
// wrappers plus the `ku_* => method` macro-table rows. Shared by the two drift guards below.
fn capi_exports() -> Vec<String> {
    let mut exports: Vec<String> = Vec::new();
    for src in CAPI_SRCS {
        // explicit wrappers: `... extern "C" fn ku_name(...`
        for seg in src.split("extern \"C\" fn ").skip(1) {
            let name = ident_prefix(seg);
            if name.starts_with("ku_") {
                exports.push(name);
            }
        }
        // macro-table rows: `ku_name => method`
        let parts: Vec<&str> = src.split("=>").collect();
        for left in &parts[..parts.len().saturating_sub(1)] {
            let name = trailing_ident(left);
            if name.starts_with("ku_") {
                exports.push(name);
            }
        }
    }
    exports.sort();
    exports.dedup();
    assert!(exports.len() > 250, "scraper found only {} exports; parsing likely broke", exports.len());
    exports
}

// Header-drift guard: every `ku_*` the cdylib exports must have a prototype in
// kurumi.h, or a cffi/cbindgen consumer (hodu-py) silently can't call it. Each export
// must appear as `ku_<name>(` in the header. Fails with the missing set.
#[test]
fn capi_header_declares_every_export() {
    const HEADER: &str = include_str!("../../../include/kurumi.h");
    let exports = capi_exports();
    let missing: Vec<&String> = exports.iter().filter(|s| !HEADER.contains(&format!("{s}("))).collect();
    assert!(missing.is_empty(), "kurumi.h is missing declarations for {} exports: {missing:?}", missing.len());
}

// Reverse drift guard: every public op BUILDER (`pub fn` on `impl Graph` under kurumi_core
// `graph/ops/**` + `graph.rs`) must have a matching `ku_<name>` C-ABI wrapper, so a new op
// can't silently ship without one (exactly how `eval_many` and `pad_mode` slipped). The
// core source tree is walked at test time (not a fixed include_str! list) so a brand-new
// file is scraped too. ALLOW lists the genuine non-op exclusions the audit found.
#[test]
fn capi_wraps_every_builder() {
    // Builders that legitimately have no `ku_<name>` export.
    const ALLOW: &[&str] = &[
        // name-aliased wrappers
        "cmp_eq", // ku_eq
        "cmp_lt", // ku_lt
        "select", // ku_where
        // graph accessors / constructor (not ops)
        "new",
        "id",
        "node",
        "dtype",
        "shape",
        // covered by another export
        "const_storage", // ku_constant
        // internal sdpa variants that `sdpa` (-> ku_sdpa) auto-selects
        "sdpa_decomposed",
        "sdpa_fused",
    ];

    fn collect_rs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                collect_rs(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }

    let exports: std::collections::HashSet<String> = capi_exports().into_iter().collect();
    let core = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../kurumi_core/src");
    let mut files = vec![core.join("graph.rs")];
    collect_rs(&core.join("graph/ops"), &mut files);

    let mut missing: Vec<String> = Vec::new();
    let mut n_builders = 0usize;
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        // `impl Graph` methods only (every `pub fn` in these files is one; verified by ALLOW
        // reconciling exactly). Free/other-impl `pub fn` would surface here as a false miss.
        for seg in src.split("pub fn ").skip(1) {
            let name = ident_prefix(seg);
            if name.is_empty() {
                continue;
            }
            n_builders += 1;
            if !ALLOW.contains(&name.as_str()) && !exports.contains(&format!("ku_{name}")) {
                missing.push(name);
            }
        }
    }
    assert!(n_builders > 200, "scraper found only {n_builders} builders; walk/parse likely broke");
    missing.sort();
    missing.dedup();
    assert!(missing.is_empty(), "op builders with no ku_* C wrapper (add one, or allowlist a non-op): {missing:?}");
}
