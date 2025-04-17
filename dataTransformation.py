"""
This script will be used to produce more training data to load into the preloaded model, with the goal of providing enough user collected data with enough playthrough to properly train the model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define global variables for use here
INPUT_DIR   = Path("/path/to/your/csvs")        # Placeholder until we get these in the repo -Preston
OUTPUT_DIR  = Path("/path/to/augmented/output") # Placeholder until we create this
NOISE_SCALE = 0.01                              # std‑dev of the perturbation
SEED        = 42                                # for reproducibility

np.random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# only tweak these columns:
TO_PERTURB = ["playerX", "playerY", "playerVelocityX", "playerVelocityY"]

for csv_path in INPUT_DIR.glob("*.csv"):
    df = pd.read_csv(csv_path)

    # apply a small random offset to each of the selected columns
    for col in TO_PERTURB:
        if col in df.columns:
            noise = np.random.normal(loc=0.0, scale=NOISE_SCALE, size=len(df))
            df[col] = df[col] + noise
        else:
            raise KeyError(f"Column '{col}' not found in {csv_path.name}")

    # write out with the same name (or add a suffix: f"{csv_path.stem}_aug.csv")
    out_path = OUTPUT_DIR / csv_path.name
    df.to_csv(out_path, index=False)

    print(f"Augmented {csv_path.name} → {out_path.name}")