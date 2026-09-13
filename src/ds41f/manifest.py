"""Run manifest: pin the source, config and placement that produced an artifact.

The review asked for reproducible evidence. A results file should be able to name
its own provenance: which engine and model revisions, which source files, which
checkpoint, which placement calibration (content hash and permutation identity),
and which ``DSV41F_*`` flags produced a number. This records that as JSON.

Deliberately cheap: checkpoint tensors are identified by size, mtime and a hash of
the safetensors header, not by hashing 130 GB per rank. Pass ``full_ckpt_hash`` only
when a run needs it and can pay for it.

CLI::

    python -m ds41f.manifest --out results/manifest.json \
        --engine-repo /root/ds41f --model-repo /root/glm-testing \
        --source src/ds41f/backend/reference.py \
        --config /root/glm-testing/ds41f/inference/config.json \
        --ckpt /data/ds41f/DSV41F-TP4 \
        --placement /root/glm-testing/ds41f/inference/expert_placement.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

__all__ = [
    "sha256_file",
    "file_identity",
    "git_identity",
    "env_snapshot",
    "runtime_versions",
    "build_manifest",
]

# Env vars that change what a run measures. Everything DSV41F_* is captured;
# these named ones are outside that prefix but still part of run identity.
_EXTRA_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_IB_DISABLE",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
)


def sha256_file(path: str | os.PathLike, limit: int | None = None) -> str:
    """sha256 of a file, or of its first ``limit`` bytes when given."""
    h = hashlib.sha256()
    remaining = limit
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return h.hexdigest()


def _safetensors_header_len(path: str | os.PathLike) -> int | None:
    """Total bytes of a safetensors header: the 8-byte size prefix plus the JSON index.

    Returns None when the file is not a plausible safetensors shard, so callers fall
    back to a fixed prefix. The header is the tensor index, so hashing all of it
    identifies the tensor layout. A fixed 1 MiB prefix does not: the shipped rank-0
    shard's header is 2,839,800 bytes, so a 1 MiB hash covers only part of the index.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            raw = fh.read(8)
    except OSError:
        return None
    if len(raw) != 8:
        return None
    n = struct.unpack("<Q", raw)[0]
    if n <= 0 or 8 + n > size:
        return None
    return 8 + n


def file_identity(path: str | os.PathLike, header_bytes: int = 1 << 20, full_hash: bool = False) -> dict:
    """Identity of one file: size, mtime, and a header hash (or the full hash).

    For a ``.safetensors`` shard the header hash covers the entire tensor index
    (8-byte length prefix plus the JSON header), which identifies the checkpoint's
    tensor layout without reading the payload. Other files hash their first
    ``header_bytes`` bytes. ``sha256_scope`` records which of these was used.
    """
    p = Path(path)
    st = p.stat()
    info = {
        "path": str(p.resolve()),
        "bytes": st.st_size,
        "mtime": st.st_mtime,
    }
    if full_hash:
        info["sha256"] = sha256_file(p)
        info["sha256_scope"] = "full"
        return info
    st_header = _safetensors_header_len(p) if p.suffix == ".safetensors" else None
    if st_header is not None:
        info["sha256"] = sha256_file(p, limit=st_header)
        info["sha256_scope"] = f"safetensors_header_{st_header}_bytes"
    else:
        info["sha256"] = sha256_file(p, limit=header_bytes)
        info["sha256_scope"] = f"first_{header_bytes}_bytes"
    return info


def _run_git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_identity(repo: str | os.PathLike) -> dict:
    """Revision and dirty state of a checkout, or ``available: false`` if not one."""
    p = Path(repo)
    revision = _run_git(p, "rev-parse", "HEAD")
    if revision is None:
        return {"path": str(p.resolve()), "available": False}
    status = _run_git(p, "status", "--porcelain") or ""
    return {
        "path": str(p.resolve()),
        "available": True,
        "revision": revision,
        "branch": _run_git(p, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines()[:50],
    }


def env_snapshot(prefixes: tuple[str, ...] = ("DSV41F_",)) -> dict:
    """Environment variables that are part of run identity, sorted for stable output."""
    keys = {k for k in os.environ if any(k.startswith(p) for p in prefixes)}
    keys.update(k for k in _EXTRA_ENV if k in os.environ)
    return {k: os.environ[k] for k in sorted(keys)}


def runtime_versions() -> dict:
    """Versions of the stack, imported lazily so a CPU manifest run stays cheap."""
    out = {"python": sys.version.split()[0]}
    for name, attr in (
        ("torch", "__version__"),
        ("triton", "__version__"),
        ("tilelang", "__version__"),
    ):
        try:
            module = __import__(name)
            out[name] = getattr(module, attr, "unknown")
        except Exception:  # noqa: BLE001 - a missing optional dep is not a failure
            out[name] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["device_count"] = torch.cuda.device_count()
            out["device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            try:
                out["nccl"] = ".".join(str(v) for v in torch.cuda.nccl.version())
            except Exception:  # noqa: BLE001
                out["nccl"] = None
    except Exception:  # noqa: BLE001
        pass
    return out


def build_manifest(
    engine_repo: str | os.PathLike | None = None,
    model_repo: str | os.PathLike | None = None,
    sources: tuple[str | os.PathLike, ...] = (),
    config: str | os.PathLike | None = None,
    ckpt: str | os.PathLike | None = None,
    placement: str | os.PathLike | None = None,
    full_ckpt_hash: bool = False,
    extra: dict | None = None,
) -> dict:
    """Collect a JSON-serializable provenance record for a run.

    ``sources`` are hashed in full (they are small). ``ckpt`` is a directory whose
    files are identified by header hash unless ``full_ckpt_hash`` is set. ``placement``
    is fingerprinted with :func:`ds41f.backend.expert_placement.fingerprint`, which
    also records whether every layer is a permutation.
    """
    manifest: dict = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": env_snapshot(),
        "runtime": runtime_versions(),
        "sources": {
            str(Path(p)): sha256_file(p) for p in sources
        },
    }
    if engine_repo is not None:
        manifest["engine_repo"] = git_identity(engine_repo)
    if model_repo is not None:
        manifest["model_repo"] = git_identity(model_repo)
    if config is not None:
        manifest["config"] = file_identity(config, full_hash=True)
    if ckpt is not None:
        ckpt_path = Path(ckpt)
        files = sorted(p for p in ckpt_path.iterdir() if p.is_file())
        manifest["checkpoint"] = {
            "path": str(ckpt_path.resolve()),
            "files": {
                p.name: file_identity(p, full_hash=full_ckpt_hash) for p in files
            },
        }
    if placement is not None:
        from .backend import expert_placement

        manifest["placement"] = expert_placement.fingerprint(str(placement))
    if extra:
        manifest["extra"] = extra
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Record a run provenance manifest as JSON.")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--engine-repo", default=None)
    ap.add_argument("--model-repo", default=None)
    ap.add_argument("--source", action="append", default=[], help="source file to hash (repeatable)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--placement", default=None)
    ap.add_argument(
        "--full-ckpt-hash",
        action="store_true",
        help="hash checkpoint shards in full instead of just their headers (slow)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_manifest(
        engine_repo=args.engine_repo,
        model_repo=args.model_repo,
        sources=tuple(args.source),
        config=args.config,
        ckpt=args.ckpt,
        placement=args.placement,
        full_ckpt_hash=args.full_ckpt_hash,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    if "placement" in manifest:
        pl = manifest["placement"]
        print(f"placement sha256={pl['sha256']} valid_permutation={pl['valid_permutation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
