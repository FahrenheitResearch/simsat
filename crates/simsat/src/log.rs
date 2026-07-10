//! Suppressible diagnostic log sink (the quiet-binding fix).
//!
//! The engine's diagnostic chatter (ingest progress / warnings, previously raw
//! `eprintln!`) goes through this sink so a LIBRARY consumer — the `simsat_py`
//! Python binding — can silence it. The sink is a process-global flag that is
//! **enabled by default**, so the CLI examples and the studio are unchanged by
//! construction; the binding disables it at module init (opt back in with
//! `SIMSAT_LOG=1` or `simsat.set_verbose(True)`).
//!
//! Semantics, by design:
//!
//! - **Process-global.** One `AtomicBool` for the whole process, matching the
//!   established once-per-process global-rayon-pool pattern
//!   (`topdown::configure_global_rayon`). Per-call verbosity plumbing was
//!   deliberately not built — the consumers are whole-process personalities
//!   (CLI/studio: chatty; imported library: quiet).
//! - **Diagnostic chatter ONLY.** The flag gates progress/warning lines on
//!   stderr. Honesty surfacing is DATA, not logs, and is untouched: the
//!   `time_is_fallback` / `GroundSource` / warning fields on
//!   `api::RenderResult` (and the Python `UserWarning`s built from them) do not
//!   route through here.
//! - **Message text unchanged.** The swapped call sites keep their exact
//!   wording (downstream scripts may parse the lines when enabled), and the
//!   lines still go to STDERR.
//! - **Never panics.** Unlike `eprintln!`, a failed stderr write is swallowed —
//!   a library must not crash its host because fd 2 is closed.

use std::sync::atomic::{AtomicBool, Ordering};

/// Whether the diagnostic sink writes. Default TRUE: the CLI examples and the
/// studio keep their existing stderr behavior without doing anything.
static ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable the engine's diagnostic stderr lines for this process.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Whether the diagnostic sink is currently enabled.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// The sink behind [`crate::log_line!`]: write one formatted line to stderr
/// when the sink is enabled. Write failures are swallowed (see module docs).
pub fn emit(args: std::fmt::Arguments<'_>) {
    write_gated(&mut std::io::stderr().lock(), enabled(), args);
}

/// The pure, testable core: write `args` (plus a newline) to `w` only when
/// `enabled`. [`emit`] passes stderr + the global flag; tests pass a buffer.
fn write_gated<W: std::io::Write>(w: &mut W, enabled: bool, args: std::fmt::Arguments<'_>) {
    if enabled {
        let _ = writeln!(w, "{args}");
    }
}

/// `eprintln!`-shaped diagnostic logging through the suppressible sink.
///
/// Same format syntax and same destination (stderr) as the `eprintln!` calls it
/// replaces; emits nothing when the sink is disabled ([`log::set_enabled`]).
///
/// [`log::set_enabled`]: crate::log::set_enabled
#[macro_export]
macro_rules! log_line {
    ($($arg:tt)*) => {
        $crate::log::emit(::std::format_args!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The global flag defaults to enabled (CLI/studio unchanged by
    /// construction) and round-trips through `set_enabled`. Kept as ONE test so
    /// the global-state mutation is serial within it (no cross-test race); no
    /// other test in the workspace asserts on sink output.
    #[test]
    fn sink_default_is_enabled_and_set_enabled_round_trips() {
        assert!(enabled(), "the sink must default ON");
        set_enabled(false);
        assert!(!enabled());
        set_enabled(true);
        assert!(enabled());
    }

    #[test]
    fn disabled_sink_writes_nothing() {
        let mut buf: Vec<u8> = Vec::new();
        write_gated(
            &mut buf,
            false,
            format_args!("simsat ingest: run=x wall=1.00s"),
        );
        assert!(buf.is_empty(), "a disabled sink must emit zero bytes");
    }

    #[test]
    fn enabled_sink_writes_the_formatted_line() {
        let mut buf: Vec<u8> = Vec::new();
        write_gated(
            &mut buf,
            true,
            format_args!("simsat ingest: run={} n={}", "abc", 42),
        );
        assert_eq!(
            String::from_utf8(buf).unwrap(),
            "simsat ingest: run=abc n=42\n"
        );
    }
}
