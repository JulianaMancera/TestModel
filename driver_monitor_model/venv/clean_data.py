import pandas as pd
import numpy as np

# ---- Load the dataset ----
df = pd.read_csv('driver_dataset.csv', header=0)

print(f"Before cleaning: {len(df)} rows")
print(df['label'].value_counts())

# ---- Remove suspicious SHA values ----
df = df[abs(df['SHA']) <= 50]
print(f"\nAfter removing bad SHA: {len(df)} rows")

# ---- Remove suspicious EAR values ----
df = df[df['EAR'] >= 0.1]
df = df[df['EAR'] <= 0.8]
print(f"After removing bad EAR: {len(df)} rows")

# ---- Remove suspicious MAR values ----
df = df[df['MAR'] >= 0.3]
df = df[df['MAR'] <= 1.5]
print(f"After removing bad MAR: {len(df)} rows")

# ---- Check final label balance ----
print("\nFinal label counts:")
print(df['label'].value_counts())

# ---- Balance the labels (equal rows per label) ----
min_count = df['label'].value_counts().min()
df_balanced = df.groupby('label').apply(
    lambda x: x.sample(min_count, random_state=42)
).reset_index(level='label').reset_index(drop=True)

print(f"\nAfter balancing ({min_count} each):")
print(df_balanced['label'].value_counts())

# ---- Shuffle the data ----
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ---- Save clean version ----
df_balanced.to_csv('driver_dataset_clean.csv', index=False)
print(f"\nSaved clean file: driver_dataset_clean.csv")
print(f"Total rows: {len(df_balanced)}")

# ---- Final stats per label ----
print("\nMean values per label:")
for label in df_balanced['label'].unique():
    sub = df_balanced[df_balanced['label'] == label]
    print(f"{label}: EAR={sub['EAR'].mean():.4f}, "
          f"MAR={sub['MAR'].mean():.4f}, "
          f"SHA={sub['SHA'].mean():.4f}")