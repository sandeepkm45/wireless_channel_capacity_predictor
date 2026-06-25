"""
generate_dataset.py
--------------------
Phase 2: Synthetic wireless channel dataset generator.

Simulates realistic wireless links and computes their Shannon channel
capacity. The dataset is built so that capacity depends NONLINEARLY on the
raw link parameters (distance, frequency, power, environment) -- the model
never sees SNR or path-loss directly as inputs, only the physical setup
that produces them. This is what makes simple models (Linear Regression)
underperform and Random Forest / Neural Network shine later.

Physics pipeline per sample:
  1. Log-distance path loss model (with reference free-space loss at 1 m)
  2. Environment-dependent path-loss exponent + log-normal shadowing
  3. Received SNR from Tx power, antenna gain, thermal noise floor
  4. Shannon-Hartley capacity: C = B * log2(1 + SNR)
  5. Small-scale fading jitter applied to the final capacity

Run from the project root:
    python src/generate_dataset.py
Output:
    data/wireless_dataset.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

N_SAMPLES = 10_000
SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "wireless_dataset.csv"

# Candidate values pulled from real cellular / Wi-Fi / mmWave bands
FREQUENCY_BANDS_GHZ = [0.9, 1.8, 2.1, 2.4, 3.5, 5.0, 28.0]      # GSM..mmWave
BANDWIDTH_OPTIONS_MHZ = [5, 10, 15, 20, 40, 80, 100, 200]
ENVIRONMENTS = ["Urban", "Suburban", "Rural"]

# Environment-dependent propagation parameters (path-loss exponent n,
# shadowing std-dev sigma in dB). Free space ~= n=2.
ENV_PARAMS = {
    "Urban":    {"n": 3.8, "sigma": 9.0},
    "Suburban": {"n": 3.2, "sigma": 7.0},
    "Rural":    {"n": 2.3, "sigma": 4.5},
}

SPEED_OF_LIGHT = 3e8  # m/s


def free_space_path_loss_db(distance_m, frequency_ghz):
    """FSPL at the reference distance, using the standard formula:
    FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
    with d in metres and f in Hz.
    """
    frequency_hz = frequency_ghz * 1e9
    return (
        20 * np.log10(distance_m)
        + 20 * np.log10(frequency_hz)
        + 20 * np.log10(4 * np.pi / SPEED_OF_LIGHT)
    )


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Raw link parameters (the model's INPUT features) ----
    distance_m = rng.uniform(10, 5000, n_samples)
    frequency_ghz = rng.choice(FREQUENCY_BANDS_GHZ, n_samples)
    bandwidth_mhz = rng.choice(BANDWIDTH_OPTIONS_MHZ, n_samples)
    tx_power_dbm = rng.uniform(20, 46, n_samples)        # typical BS/AP Tx power
    antenna_gain_db = rng.uniform(0, 20, n_samples)
    noise_figure_db = rng.uniform(5, 10, n_samples)
    environment = rng.choice(ENVIRONMENTS, n_samples)

    # ---- Physics: path loss (log-distance model w/ shadowing) ----
    n_exp = np.array([ENV_PARAMS[e]["n"] for e in environment])
    sigma = np.array([ENV_PARAMS[e]["sigma"] for e in environment])

    d0 = 1.0  # reference distance in metres
    pl_d0 = free_space_path_loss_db(d0, frequency_ghz)
    shadowing_db = rng.normal(0, sigma)
    path_loss_db = pl_d0 + 10 * n_exp * np.log10(distance_m / d0) + shadowing_db

    # ---- SNR ----
    received_power_dbm = tx_power_dbm + antenna_gain_db - path_loss_db
    noise_power_dbm = -174 + 10 * np.log10(bandwidth_mhz * 1e6) + noise_figure_db
    snr_db = received_power_dbm - noise_power_dbm
    snr_linear = 10 ** (snr_db / 10)

    # ---- Shannon-Hartley capacity ----
    capacity_mbps = bandwidth_mhz * np.log2(1 + snr_linear)

    # ---- Small-scale fading jitter (Rayleigh-ish multiplicative noise) ----
    fading_factor = rng.uniform(0.85, 1.15, n_samples)
    capacity_mbps = capacity_mbps * fading_factor
    capacity_mbps = np.clip(capacity_mbps, 0, None)

    df = pd.DataFrame({
        "distance_m": distance_m.round(2),
        "frequency_ghz": frequency_ghz,
        "bandwidth_mhz": bandwidth_mhz,
        "tx_power_dbm": tx_power_dbm.round(2),
        "antenna_gain_db": antenna_gain_db.round(2),
        "noise_figure_db": noise_figure_db.round(2),
        "environment": environment,
        "capacity_mbps": capacity_mbps.round(3),
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df):,} samples -> {OUTPUT_PATH}")
    print("\nPreview:")
    print(df.head())
    print("\nCapacity (Mbps) summary:")
    print(df["capacity_mbps"].describe().round(2))
    print("\nEnvironment distribution:")
    print(df["environment"].value_counts())