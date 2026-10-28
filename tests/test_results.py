import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())


def test_batching_always_helps_sdxl():
    assert S["sdxl_gain_min"] > 1.0 and S["sdxl_gain_max"] > 2.0


def test_ltx_load_time_dwarfs_generation_time():
    assert S["ltx_load_over_gen"] > 50


def test_gb10_is_the_slowest_flux_host():
    assert S["flux_gb10_s"] > 3 * S["flux_best_s"]


def test_flux_fp8_needs_ada_or_newer():
    assert S["flux_failed_gpu"] == "RTX 3090"


def test_artifacts_are_present():
    for f in ("sdxl_bs1.png", "sdxl_bs4.png", "ltx_25.mp4"):
        assert (ROOT / "artifacts" / f).stat().st_size > 10_000
