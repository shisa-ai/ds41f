"""Run manifest: provenance pinning for source, config and placement.

The review asked for a manifest that pins a run's identity, including the placement
calibration's content hash and permutation validity, so a results file can name its
own provenance instead of relying on "the file next to the module".
"""

import hashlib
import json
import shutil
import subprocess

import pytest
import torch

from ds41f import manifest
from ds41f.backend import expert_placement as ep


def _write_placement(path, layers=3, experts=6):
    n2o = torch.stack([torch.randperm(experts) for _ in range(layers)])
    torch.save({"new_to_old": n2o, "loads": torch.rand(layers, experts)}, path)
    return path


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    payload = b"ds41f" * 1000
    p.write_bytes(payload)
    assert manifest.sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_file_identity_header_scope_is_not_the_full_hash(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"a" * (1 << 20) + b"b" * 4096)
    header = manifest.file_identity(p, header_bytes=1 << 20)
    full = manifest.file_identity(p, full_hash=True)
    assert header["sha256_scope"] == "first_1048576_bytes"
    assert full["sha256_scope"] == "full"
    assert header["bytes"] == full["bytes"] == (1 << 20) + 4096
    assert header["sha256"] != full["sha256"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_identity_tracks_revision_and_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "f.txt").write_text("one\n")
    git("add", "f.txt")
    git("commit", "-q", "-m", "one")

    clean = manifest.git_identity(repo)
    assert clean["available"] is True
    assert len(clean["revision"]) == 40
    assert clean["dirty"] is False

    (repo / "f.txt").write_text("two\n")
    dirty = manifest.git_identity(repo)
    assert dirty["dirty"] is True
    assert "f.txt" in dirty["dirty_paths"][0]


def test_git_identity_reports_unavailable_outside_a_repo(tmp_path):
    info = manifest.git_identity(tmp_path)
    assert info["available"] is False


def test_env_snapshot_picks_ds41f_prefix_and_named_vars(monkeypatch):
    monkeypatch.setenv("DSV41F_FP8_GEMV_MIN", "0")
    monkeypatch.setenv("DSV41F_W13_WARPS", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("UNRELATED", "x")
    snap = manifest.env_snapshot()
    assert snap["DSV41F_FP8_GEMV_MIN"] == "0"
    assert snap["DSV41F_W13_WARPS"] == "8"
    assert snap["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "UNRELATED" not in snap
    assert list(snap) == sorted(snap)


def test_fingerprint_records_hash_shape_and_validity(tmp_path):
    path = _write_placement(tmp_path / "p.pt", layers=4, experts=8)
    info = ep.fingerprint(str(path))
    assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info["layers"] == 4 and info["experts"] == 8
    assert info["valid_permutation"] is True
    assert info["invalid_layers"] == []


def test_fingerprint_flags_a_non_permutation_layer(tmp_path):
    n2o = torch.stack([torch.randperm(6), torch.tensor([0, 0, 1, 2, 3, 4])])
    path = tmp_path / "bad.pt"
    torch.save({"new_to_old": n2o}, path)
    info = ep.fingerprint(str(path))
    assert info["valid_permutation"] is False
    assert info["invalid_layers"] == [1]


def test_fingerprint_survives_a_cuda_default_device(tmp_path):
    """The serving loader sets the default device to CUDA before applying placement,
    while the artifact loads to CPU. A default-device arange made fingerprint raise
    with a device mismatch instead of validating the permutation."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    path = _write_placement(tmp_path / "p.pt", layers=2, experts=4)
    prev = torch.get_default_device()
    try:
        torch.set_default_device("cuda")
        info = ep.fingerprint(str(path))
    finally:
        torch.set_default_device(prev)
    assert info["valid_permutation"] is True
    assert info["experts"] == 4


def test_check_expected_hash_accepts_match_and_rejects_mismatch(tmp_path):
    path = _write_placement(tmp_path / "p.pt")
    info = ep.fingerprint(str(path))
    ep.check_expected_hash(info, info["sha256"])  # no raise
    ep.check_expected_hash(info, None)  # unset pin: accept
    with pytest.raises(ValueError, match="does not match"):
        ep.check_expected_hash(info, "0" * 64)


def test_expected_sha256_reads_and_normalizes_env(monkeypatch):
    monkeypatch.delenv("DSV41F_EXPERT_PLACEMENT_SHA256", raising=False)
    assert ep.expected_sha256() is None
    monkeypatch.setenv("DSV41F_EXPERT_PLACEMENT_SHA256", "  ABCDEF  ")
    assert ep.expected_sha256() == "abcdef"
    monkeypatch.setenv("DSV41F_EXPERT_PLACEMENT_SHA256", "   ")
    assert ep.expected_sha256() is None


def test_build_manifest_records_every_requested_section(tmp_path, monkeypatch):
    monkeypatch.setenv("DSV41F_ENGRAM_OFFLOAD", "1")
    src = tmp_path / "reference.py"
    src.write_text("x = 1\n")
    cfg = tmp_path / "config.json"
    cfg.write_text('{"vocab_size": 1}\n')
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model0-mp4.safetensors").write_bytes(b"header" + b"0" * 2048)
    placement = _write_placement(tmp_path / "expert_placement.pt")

    m = manifest.build_manifest(
        engine_repo=tmp_path,  # not a git repo: records available=False, must not raise
        model_repo=tmp_path,
        sources=(src,),
        config=cfg,
        ckpt=ckpt,
        placement=placement,
    )
    assert m["engine_repo"]["available"] is False
    assert m["sources"][str(src)] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert m["config"]["sha256_scope"] == "full"
    assert "model0-mp4.safetensors" in m["checkpoint"]["files"]
    assert m["checkpoint"]["files"]["model0-mp4.safetensors"]["sha256_scope"].startswith("first_")
    assert m["placement"]["valid_permutation"] is True
    assert m["env"]["DSV41F_ENGRAM_OFFLOAD"] == "1"
    # must be JSON-serializable as written
    json.dumps(m, sort_keys=True)


def test_manifest_cli_writes_json(tmp_path, monkeypatch):
    monkeypatch.setenv("DSV41F_GEMV_BLOCK_N", "8")
    placement = _write_placement(tmp_path / "expert_placement.pt")
    out = tmp_path / "nested" / "manifest.json"
    rc = manifest.main(["--out", str(out), "--placement", str(placement)])
    assert rc == 0
    written = json.loads(out.read_text())
    assert written["placement"]["experts"] == 6
    assert written["env"]["DSV41F_GEMV_BLOCK_N"] == "8"
