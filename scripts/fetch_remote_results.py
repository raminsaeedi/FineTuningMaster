"""Copy and verify a packaged result archive from a remote GPU server.

This script is run on the local PC after the remote experiment finishes. It
uses the system OpenSSH ``scp`` executable, so it adds no project dependency.

Example (PowerShell or a Unix shell)::

    python scripts/fetch_remote_results.py \
        user@h100:/mnt/big/packages/professor_results_dashboard_v4.zip \
        --destination .\\remote-results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and verify a professor result ZIP from a remote server."
    )
    parser.add_argument(
        "remote_archive",
        help="scp source, for example user@h100:/mnt/big/packages/professor_results_dashboard_v4.zip",
    )
    parser.add_argument(
        "--destination",
        default="remote-results",
        help="Local directory for the ZIP and manifest (default: remote-results)",
    )
    return parser.parse_args(argv)


def _split_remote(source: str) -> tuple[str, str]:
    if ":" not in source:
        raise ValueError("remote_archive must use scp syntax: user@host:/absolute/path/file.zip")
    host, remote_path = source.split(":", 1)
    if not host or not remote_path:
        raise ValueError("remote_archive must include both host and remote path")
    return host, remote_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scp(scp: str, source: str, target: Path) -> None:
    temporary = target.with_name(target.name + ".part")
    if temporary.exists():
        temporary.unlink()
    subprocess.run([scp, source, str(temporary)], check=True)
    temporary.replace(target)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        host, remote_path = _split_remote(args.remote_archive)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    scp = shutil.which("scp")
    if scp is None:
        raise SystemExit("scp not found. Install/enable OpenSSH on this PC and retry.")

    archive_name = PurePosixPath(remote_path).name
    if not archive_name.endswith(".zip"):
        raise SystemExit(f"Remote file must be a .zip archive: {archive_name}")
    manifest_name = f"{Path(archive_name).stem}_manifest.json"
    remote_dir = posixpath.dirname(remote_path.rstrip("/")) or "."
    remote_manifest = f"{host}:{posixpath.join(remote_dir, manifest_name)}"

    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / archive_name
    manifest_path = destination / manifest_name

    print(f"Fetching {args.remote_archive}")
    _scp(scp, args.remote_archive, archive)
    _scp(scp, remote_manifest, manifest_path)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(manifest["archive_sha256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Downloaded manifest is invalid: {exc}") from exc
    actual = _sha256(archive)
    if actual != expected:
        raise SystemExit(
            f"SHA-256 mismatch for {archive}: expected {expected}, got {actual}"
        )

    print(f"Saved: {archive}")
    print(f"Manifest: {manifest_path}")
    print(f"SHA-256: {actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
