/// Returns rather than `expect`s, because `[lints.clippy] expect_used` reaches the build script
/// too: the Windows lint job compiles this file, and the panicking call was a hard error there
/// while every other platform cfg'd the block away and never saw it.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(windows)]
    {
        let mut resource = winres::WindowsResource::new();
        resource.set_icon("assets/meka.ico");
        resource.compile()?;
    }
    Ok(())
}
