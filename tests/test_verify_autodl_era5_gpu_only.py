from __future__ import annotations

from scripts.verify_autodl_era5_gpu_only import GPU_MANIFEST_POLICIES, exclusion_reason


def test_long_batch_rows_are_explicit_rtx4090_capacity_exclusions() -> None:
    reason = exclusion_reason(
        {"method": "official_st_svgp_ms64", "device_class": "a100_official_full"},
        GPU_MANIFEST_POLICIES["official_long_full"],
    )
    assert reason == "rtx4090_long_batch_workspace_oom_ms32_requires_52_11_gib"
