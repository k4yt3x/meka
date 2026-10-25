/// Returns rather than `expect`s, because `[lints.clippy] expect_used` reaches the build script
/// too: the Windows lint job compiles this file, and the panicking call was a hard error there
/// while every other platform cfg'd the block away and never saw it.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    embed_git_revision();

    #[cfg(windows)]
    {
        let mut resource = winres::WindowsResource::new();
        resource.set_icon("assets/meka.ico");
        resource.compile()?;
    }
    Ok(())
}

/// Hand the binary the revision it was built from as `MEKA_GIT_REVISION`, which the screen's
/// banner shows beside the version: the short commit, `-dirty` when the tree had uncommitted
/// changes, so a build from a checkout can be told from the release of the same version. A
/// build outside a checkout, or without `git`, gets no value and the banner shows the version
/// alone; a packager who knows the revision sets the variable in the build's environment.
///
/// The script reruns when the sources or the checkout's head change, which is when the answer
/// can: a change to a file the binary is not built from leaves `-dirty` as it was.
fn embed_git_revision() {
    println!("cargo:rerun-if-env-changed=MEKA_GIT_REVISION");
    for watched in ["src", "tests", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={watched}");
    }
    if std::path::Path::new(".git/HEAD").is_file() {
        println!("cargo:rerun-if-changed=.git/HEAD");
        println!("cargo:rerun-if-changed=.git/index");
    }
    let given = std::env::var("MEKA_GIT_REVISION")
        .ok()
        .map(|revision| revision.trim().to_string())
        .filter(|revision| !revision.is_empty());
    let revision = given.or_else(|| {
        // `--exclude=*` leaves no tag to describe against, so `--always` answers with the short
        // commit. A clean tree at a tag is the release of that tag, which the version line
        // already says, so it gets no revision.
        let described = git(["describe", "--always", "--dirty", "--exclude=*"])?;
        let at_tag = !described.ends_with("-dirty")
            && git(["describe", "--tags", "--exact-match", "HEAD"]).is_some();
        (!at_tag).then_some(described)
    });
    if let Some(revision) = revision {
        println!("cargo:rustc-env=MEKA_GIT_REVISION={revision}");
    }
}

/// `git <args>`'s trimmed output, `None` when git is absent, the command fails or says nothing.
fn git<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!text.is_empty()).then_some(text)
}
