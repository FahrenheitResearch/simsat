//! Build-script helper for embedding Windows PE version information.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Compile one version resource and link it only into the named binary targets.
///
/// Cargo provides the version components from the owning package manifest, so a
/// normal workspace version bump automatically updates both the numeric PE
/// version and the strings shown by Windows Explorer.
pub fn embed(bin_names: &[&str], product_name: &str, file_description: &str) {
    println!("cargo:rerun-if-env-changed=RC");
    println!("cargo:rerun-if-env-changed=RC_PATH");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ENV").as_deref(),
        Ok("msvc"),
        "SimSat Windows release binaries require the MSVC target"
    );

    let version = env::var("CARGO_PKG_VERSION").expect("Cargo package version");
    let major = version_component("CARGO_PKG_VERSION_MAJOR");
    let minor = version_component("CARGO_PKG_VERSION_MINOR");
    let patch = version_component("CARGO_PKG_VERSION_PATCH");
    let product_name = rc_string(product_name);
    let file_description = rc_string(file_description);
    let version_string = rc_string(&version);

    let resource = format!(
        r#"#pragma code_page(65001)
1 VERSIONINFO
 FILEVERSION {major},{minor},{patch},0
 PRODUCTVERSION {major},{minor},{patch},0
 FILEFLAGSMASK 0x3fL
 FILEFLAGS 0x0L
 FILEOS 0x40004L
 FILETYPE 0x1L
 FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904b0"
        BEGIN
            VALUE "FileDescription", "{file_description}\0"
            VALUE "FileVersion", "{version_string}\0"
            VALUE "ProductName", "{product_name}\0"
            VALUE "ProductVersion", "{version_string}\0"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x0409, 1200
    END
END
"#
    );

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let rc_path = out_dir.join("simsat-version.rc");
    let res_path = out_dir.join("simsat-version.res");
    fs::write(&rc_path, resource).expect("write Windows version resource");

    let mut command = resource_compiler();
    let status = command
        .arg("/nologo")
        .arg(format!("/fo{}", res_path.display()))
        .arg(&rc_path)
        .status()
        .expect("run Windows resource compiler");
    assert!(
        status.success(),
        "Windows resource compiler failed: {status}"
    );

    for bin_name in bin_names {
        println!("cargo:rustc-link-arg-bin={bin_name}={}", res_path.display());
    }
}

fn version_component(name: &str) -> u16 {
    env::var(name)
        .unwrap_or_else(|_| panic!("Cargo did not set {name}"))
        .parse()
        .unwrap_or_else(|_| panic!("{name} is not a 16-bit integer"))
}

fn rc_string(value: &str) -> String {
    assert!(
        !value.contains(['\0', '\r', '\n']),
        "Windows version strings cannot contain control characters"
    );
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn resource_compiler() -> Command {
    if let Some(path) = env::var_os("RC_PATH").or_else(|| env::var_os("RC")) {
        return Command::new(path);
    }
    let target = env::var("TARGET").expect("Cargo target triple");
    find_msvc_tools::find(&target, "rc.exe")
        .unwrap_or_else(|| panic!("could not find rc.exe in the installed Windows SDK"))
}
