"""
This script will be used to produce more training data to load into the preloaded model, with the goal of providing enough user collected data with enough playthrough to properly train the model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

 # Global var definitions
 # Pointing these to newly created directory -Preston
INPUT_CSV   = Path("dataEngineering/100Attempts-corruptedSickles.csv")
OUTPUT_CSV  = Path("dataEngineering/output_augmented_100_attempts.csv")
NOISE_SCALE = 0.01    # σ of Gaussian jitter
REPS        = 10      # how many augmented copies per original row, we want 10, but can further augment if we need more data -Preston
SEED        = 42

# Which columns to jitter
TO_PERTURB = ["playerX", "playerY", "playerVelocityX", "playerVelocityY"]

np.random.seed(SEED)

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Repeat each row 10 times
df_rep = pd.concat([df]*REPS, ignore_index=True)

# Generate and add noise for each perturbed column to slightly vary the data & add to the overall dataset for training
n = len(df_rep)
for col in TO_PERTURB:
    if col not in df_rep.columns:
        raise KeyError(f"Column '{col}' not in input")
    noise = np.random.normal(loc=0.0, scale=NOISE_SCALE, size=n)
    df_rep[col] = df_rep[col] + noise

# 4) Write out the expanded, jittered dataset
df_rep.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {n} rows ({REPS}× original) to {OUTPUT_CSV}")