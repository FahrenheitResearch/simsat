#[path = "../../build-support/windows_version.rs"]
mod windows_version;

fn main() {
    windows_version::embed(
        &["simsat_studio"],
        "SimSat Studio",
        "SimSat Studio satellite simulator",
    );
}
