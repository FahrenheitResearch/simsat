//! Cross-platform ingest instrumentation: peak resident-set size and the
//! below-normal worker-thread priority (design doc section 2; peak-RSS discipline).
//!
//! Peak RSS: Linux reads `VmHWM` from `/proc/self/status`; Windows reads
//! `PeakWorkingSetSize` via `K32GetProcessMemoryInfo`. Both are the whole-process
//! high-water mark, which is exactly the number the < 2.5 GB ingest contract is
//! written against. Other platforms return `None`.
//!
//! Thread priority: `lower_worker_thread_priority` is a straight port of
//! `crates/app_ui/src/wrf_process.rs::lower_import_thread_priority` (BowEcho,
//! `THREAD_PRIORITY_BELOW_NORMAL`) — the owner's machine has hard-crashed under
//! all-core memory-bandwidth load, so every grinding worker (ingest, render,
//! rayon pool threads) yields to the desktop. Windows-only; a no-op on the
//! headless Linux build nodes so tests pass. `lower_ingest_thread_priority` is
//! the original name, kept as a delegating alias for existing call sites.

/// Parse the `VmHWM:` (peak resident set) line out of a `/proc/self/status`
/// dump and return kilobytes. Pure string logic so it is testable on any OS.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn parse_vmhwm_kb(status: &str) -> Option<u64> {
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest.split_whitespace().next()?.parse::<u64>().ok();
        }
    }
    None
}

/// Peak resident-set size of this process in bytes, or `None` if unavailable.
#[cfg(target_os = "linux")]
pub fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_vmhwm_kb(&status).map(|kb| kb.saturating_mul(1024))
}

/// Peak working-set size of this process in bytes, or `None` if unavailable.
#[cfg(windows)]
pub fn peak_rss_bytes() -> Option<u64> {
    // PROCESS_MEMORY_COUNTERS (psapi.h). `SIZE_T` == usize on the target. The
    // K32-prefixed forwarder is exported from kernel32, so no psapi link is needed.
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    let mut counters: ProcessMemoryCounters = unsafe { core::mem::zeroed() };
    counters.cb = core::mem::size_of::<ProcessMemoryCounters>() as u32;
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok != 0 {
        Some(counters.peak_working_set_size as u64)
    } else {
        None
    }
}

/// Fallback for platforms without a supported peak-RSS query.
#[cfg(not(any(target_os = "linux", windows)))]
pub fn peak_rss_bytes() -> Option<u64> {
    None
}

/// Lower the calling thread to below-normal priority (Windows only; no-op elsewhere).
///
/// Ported pattern: `crates/app_ui/src/wrf_process.rs::lower_import_thread_priority`
/// (BowEcho). Raw `kernel32` bindings are used instead of the `windows-sys` crate
/// so no new dependency is pulled in; the Linux build nodes compile the no-op stub.
///
/// The general name: this is used for ANY grinding background worker thread —
/// the ingest worker, the studio render worker, and every rayon pool thread (the
/// studio installs it as the global-pool `start_handler`) — so all-core CPU work
/// always yields to the desktop (the owner's machine has hard-crashed under
/// all-core load; machine-stability discipline, hard rule 4 spirit).
#[cfg(windows)]
pub fn lower_worker_thread_priority() {
    const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(thread: isize, priority: i32) -> i32;
    }
    unsafe {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

/// No-op on non-Windows targets (headless Linux nodes, etc.).
#[cfg(not(windows))]
pub fn lower_worker_thread_priority() {}

/// The original (ingest-specific) name, kept as a delegating alias so existing
/// call sites (`ingest.rs`, the render examples) keep compiling unchanged.
pub fn lower_ingest_thread_priority() {
    lower_worker_thread_priority();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_vmhwm_extracts_peak_kilobytes() {
        let sample =
            "Name:\tsimsat\nVmPeak:\t  204800 kB\nVmHWM:\t   123456 kB\nVmRSS:\t  100000 kB\n";
        assert_eq!(parse_vmhwm_kb(sample), Some(123_456));
    }

    #[test]
    fn parse_vmhwm_absent_returns_none() {
        assert_eq!(parse_vmhwm_kb("Name:\tsimsat\nVmRSS:\t 100 kB\n"), None);
        assert_eq!(parse_vmhwm_kb(""), None);
    }

    #[test]
    fn lower_priority_never_panics() {
        // On Linux this is a stub; on Windows it flips the current thread. Either
        // way it must be safe to call from a test.
        lower_ingest_thread_priority();
    }

    #[test]
    fn lower_worker_priority_never_panics() {
        // The general worker name (the ingest name delegates to it) must also be
        // safe to call anywhere — it is installed as a rayon pool start_handler.
        lower_worker_thread_priority();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn peak_rss_is_available_on_linux() {
        let rss = peak_rss_bytes().expect("VmHWM present on linux");
        assert!(rss > 0, "peak rss should be positive, got {rss}");
    }
}
