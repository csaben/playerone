//! Push a finished session shard to a Hugging Face dataset repo via the `hf` CLI.
//!
//! Shelling out to `hf upload` (huggingface_hub 1.x) rather than reimplementing the commit API —
//! same pragmatic call rlrec made. The command is configurable so callers can use `hf` directly or
//! `uv run hf` (this machine has hf only inside a uv env). Auth comes from the stored HF token.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

#[derive(Clone, Debug)]
pub struct UploadConfig {
    /// Target dataset repo, e.g. "arelius/rlrec-desktop-capture".
    pub repo: String,
    /// Folder (path within the repo) to upload the session under. `None`/empty → the session dir's
    /// own name, so each session still lands under its own folder.
    pub folder: Option<String>,
    /// The CLI invocation, e.g. ["hf"] or ["uv","run","hf"].
    pub hf_cmd: Vec<String>,
    /// Working directory (needed when hf_cmd is `uv run hf` inside a uv project).
    pub cwd: Option<PathBuf>,
}

/// `hf upload <repo> <local_dir> <path_in_repo> --repo-type dataset`. `folder` (if non-empty) is the
/// path within the repo; otherwise the session dir's name is used, so each session lands under its
/// own folder.
pub fn build_args(session_dir: &Path, repo: &str, folder: Option<&str>) -> Vec<String> {
    let path_in_repo = folder
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .map(|f| f.to_string())
        .unwrap_or_else(|| {
            session_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "session".into())
        });
    vec![
        "upload".into(),
        repo.to_string(),
        session_dir.display().to_string(),
        path_in_repo,
        "--repo-type".into(),
        "dataset".into(),
    ]
}

pub fn upload(session_dir: &Path, cfg: &UploadConfig) -> Result<()> {
    ensure_complete(session_dir)?;
    let (bin, pre) = cfg.hf_cmd.split_first().context("empty hf_cmd")?;
    let mut cmd = Command::new(bin);
    cmd.args(pre).args(build_args(session_dir, &cfg.repo, cfg.folder.as_deref()));
    if let Some(cwd) = &cfg.cwd {
        cmd.current_dir(cwd);
    }
    let status = cmd
        .status()
        .with_context(|| format!("spawn hf ({})", cfg.hf_cmd.join(" ")))?;
    if !status.success() {
        bail!("hf upload exited with {status}");
    }
    Ok(())
}

fn ensure_complete(session_dir: &Path) -> Result<()> {
    if !session_dir.join(super::COMPLETE_NAME).is_file() {
        bail!("refusing to upload incomplete recording {} (missing {})", session_dir.display(), super::COMPLETE_NAME);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_put_session_name_as_repo_path() {
        let args = build_args(Path::new("/scratch/s123-abc"), "user/ds", None);
        assert_eq!(args[0], "upload");
        assert_eq!(args[1], "user/ds");
        assert_eq!(args[2], "/scratch/s123-abc");
        assert_eq!(args[3], "s123-abc");
        assert_eq!(&args[4..], &["--repo-type", "dataset"]);
    }

    #[test]
    fn explicit_folder_overrides_session_name() {
        let args = build_args(Path::new("/scratch/s123-abc"), "user/ds", Some("Notepad-20260725-014200"));
        assert_eq!(args[3], "Notepad-20260725-014200");
        let blank = build_args(Path::new("/scratch/s123-abc"), "user/ds", Some("  "));
        assert_eq!(blank[3], "s123-abc"); // blank folder falls back to the session name
    }

    #[test]
    fn incomplete_session_is_never_publishable() {
        let dir = std::env::temp_dir().join(format!("agentpc_upload_test_{}", super::super::default_session_id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(ensure_complete(&dir).unwrap_err().to_string().contains("incomplete recording"));
        std::fs::write(dir.join(super::super::COMPLETE_NAME), b"{}").unwrap();
        ensure_complete(&dir).unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }
}
