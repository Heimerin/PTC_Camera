#!/usr/bin/env python3
"""Generate small synthetic NPY PTC data for smoke-testing ptc_analyzer.py."""

from pathlib import Path
import shutil
import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)
    root = Path("synthetic_data")
    if root.exists():
        shutil.rmtree(root)
    bias_dir = root / "bias"
    flat_dir = root / "flats"
    bias_dir.mkdir(parents=True)
    flat_dir.mkdir(parents=True)

    shape = (180, 240)
    offset = 500.0
    read_noise_dn = 3.0
    k_e_per_dn = 1.7
    prnu = rng.normal(1.0, 0.006, size=shape)

    for i in range(12):
        bias = offset + rng.normal(0, read_noise_dn, size=shape)
        np.save(bias_dir / f"bias_{i:03d}.npy", bias.astype(np.float32))

    exposures = np.geomspace(0.001, 1.0, 18)
    photons_per_second_e = 85000.0
    for exp in exposures:
        signal_e = photons_per_second_e * exp
        # Mild compression near the top, just to create a visible nonlinearity.
        signal_e = signal_e * (1.0 - 0.08 * (signal_e / 85000.0) ** 2)
        signal_dn = signal_e / k_e_per_dn
        for j in range(2):
            shot_dn = rng.normal(0, np.sqrt(max(signal_e, 1.0)) / k_e_per_dn, size=shape)
            read = rng.normal(0, read_noise_dn, size=shape)
            flat = offset + signal_dn * prnu + shot_dn + read
            unit = "ms"
            exp_ms = exp * 1000.0
            np.save(flat_dir / f"flat_{exp_ms:.3f}{unit}_{j+1}.npy", flat.astype(np.float32))

    print(root.resolve())


if __name__ == "__main__":
    main()
