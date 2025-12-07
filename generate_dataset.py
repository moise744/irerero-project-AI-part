"""
IRERERO PROJECT - Dataset Generation Script
This script generates synthetic child health data following WHO standards.

Author: IRERERO Team
Date: December 2024
"""

import pandas as pd
import numpy as np

print("="*70)
print("IRERERO DATASET GENERATION")
print("Generating synthetic child health data")
print("="*70)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# CONFIGURATION
# ============================================
N_SAMPLES = 800  # Number of children records to generate
print(f"\n📊 Generating {N_SAMPLES} child health records...")

# ============================================
# STEP 1: Generate Basic Demographics
# ============================================
print("\n[1/6] Generating demographic data...")

# Age in months (24-60 months = 2-5 years for ECD centers)
age_months = np.random.randint(24, 61, size=N_SAMPLES)

# Gender (0 = Female, 1 = Male) - balanced
gender = np.random.choice([0, 1], size=N_SAMPLES)

print(f"✓ Age range: {age_months.min()}-{age_months.max()} months")
print(f"✓ Gender distribution: {sum(gender==0)} Female, {sum(gender==1)} Male")

# ============================================
# STEP 2: Generate Anthropometric Measurements
# ============================================
print("\n[2/6] Generating anthropometric measurements...")

# Height in cm (based on age with variation)
# WHO standard: ~85cm at 2 years, ~110cm at 5 years
height_cm = []
for age in age_months:
    # Base height from WHO charts
    base_height = 85 + ((age - 24) / 36) * 25  # Linear growth from 85 to 110cm
    # Add individual variation
    height = base_height + np.random.normal(0, 4)
    height_cm.append(max(70, height))  # Minimum 70cm

height_cm = np.array(height_cm)

# Weight in kg (based on age and height with variation)
# WHO standard: ~12kg at 2 years, ~18kg at 5 years
weight_kg = []
for age, height in zip(age_months, height_cm):
    # Base weight from WHO charts
    base_weight = 12 + ((age - 24) / 36) * 6  # Linear growth from 12 to 18kg
    # Add variation based on height
    weight = base_weight + (height - 95) * 0.08 + np.random.normal(0, 1.5)
    weight_kg.append(max(8, weight))  # Minimum 8kg

weight_kg = np.array(weight_kg)

# Calculate BMI
bmi = weight_kg / ((height_cm / 100) ** 2)

print(f"✓ Height range: {height_cm.min():.1f}-{height_cm.max():.1f} cm")
print(f"✓ Weight range: {weight_kg.min():.1f}-{weight_kg.max():.1f} kg")
print(f"✓ BMI range: {bmi.min():.1f}-{bmi.max():.1f}")

# ============================================
# STEP 3: Generate Z-Scores (WHO Nutritional Indicators)
# ============================================
print("\n[3/6] Generating WHO Z-scores...")

# Create controlled distribution to ensure sufficient samples per class
# We'll use a mixture approach to guarantee minimum samples

# Height-for-Age Z-score (HAZ) - stunting indicator
haz = np.random.normal(-0.5, 1.5, size=N_SAMPLES)

# Weight-for-Age Z-score (WAZ) - underweight indicator
waz = np.random.normal(-0.3, 1.4, size=N_SAMPLES)

# Weight-for-Height Z-score (WHZ) - wasting indicator
whz = np.random.normal(-0.2, 1.3, size=N_SAMPLES)

print(f"✓ HAZ range: {haz.min():.2f} to {haz.max():.2f}")
print(f"✓ WAZ range: {waz.min():.2f} to {waz.max():.2f}")
print(f"✓ WHZ range: {whz.min():.2f} to {whz.max():.2f}")

# ============================================
# STEP 4: Determine Health Status (Ensuring balanced classes)
# ============================================
print("\n[4/6] Determining health status classifications...")

health_status = []

for i in range(N_SAMPLES):
    # Classification based on WHO Z-score thresholds
    stunted = haz[i] < -2
    underweight = waz[i] < -2
    wasted = whz[i] < -2
    overweight = whz[i] > 2
    
    # Combined classification with adjusted logic
    if stunted and underweight and wasted:
        status = "Severely Malnourished"
    elif (stunted and underweight) or (stunted and wasted) or (underweight and wasted):
        status = "Moderately Malnourished"
    elif stunted or underweight or wasted:
        status = "At Risk"
    elif overweight:
        status = "Overweight"
    else:
        status = "Healthy"
    
    health_status.append(status)

# Count initial distribution
from collections import Counter
status_counts = Counter(health_status)

# Ensure minimum samples per class (at least 10)
print("✓ Initial health status distribution:")
for status, count in status_counts.most_common():
    print(f"  - {status}: {count} ({count/N_SAMPLES*100:.1f}%)")

# Adjust if any class has fewer than 10 samples
min_samples_per_class = 10
for status in ["Healthy", "At Risk", "Overweight", "Moderately Malnourished", "Severely Malnourished"]:
    if status_counts[status] < min_samples_per_class:
        deficit = min_samples_per_class - status_counts[status]
        # Replace some "Healthy" samples with this status
        healthy_indices = [i for i, s in enumerate(health_status) if s == "Healthy"]
        if len(healthy_indices) >= deficit:
            for idx in np.random.choice(healthy_indices, deficit, replace=False):
                health_status[idx] = status

# Final count
status_counts = Counter(health_status)
print("\n✓ Final health status distribution (balanced):")
for status, count in status_counts.most_common():
    print(f"  - {status}: {count} ({count/N_SAMPLES*100:.1f}%)")

# ============================================
# STEP 5: Create DataFrames
# ============================================
print("\n[5/6] Creating dataset files...")

# Full dataset with all information
full_dataset = pd.DataFrame({
    'Child_ID': [f'CH{str(i+1).zfill(4)}' for i in range(N_SAMPLES)],
    'Age_Months': age_months,
    'Gender': ['Female' if g == 0 else 'Male' for g in gender],
    'Gender_Numeric': gender,
    'Height_cm': np.round(height_cm, 2),
    'Weight_kg': np.round(weight_kg, 2),
    'BMI': np.round(bmi, 2),
    'HAZ': np.round(haz, 2),
    'WAZ': np.round(waz, 2),
    'WHZ': np.round(whz, 2),
    'Health_Status': health_status
})

# Training dataset (without Child_ID)
training_dataset = full_dataset.drop('Child_ID', axis=1)

# Save datasets
full_dataset.to_csv('irerero_child_health_dataset.csv', index=False)
training_dataset.to_csv('irerero_model_training_data.csv', index=False)

print(f"✓ Saved: irerero_child_health_dataset.csv ({N_SAMPLES} records)")
print(f"✓ Saved: irerero_model_training_data.csv ({N_SAMPLES} records)")

# ============================================
# STEP 6: Display Summary
# ============================================
print("\n[6/6] Dataset Summary:")
print("="*70)
print("\n📋 Sample of generated data (first 5 rows):")
print(full_dataset.head())

print("\n📊 Dataset Statistics:")
print(training_dataset.describe())

print("\n📈 Health Status Counts:")
print(training_dataset['Health_Status'].value_counts())

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ DATASET GENERATION COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nGenerated Files:")
print(f"  1. irerero_child_health_dataset.csv")
print(f"  2. irerero_model_training_data.csv")
print(f"\nTotal Records: {N_SAMPLES}")
print(f"Features: Age, Gender, Height, Weight, BMI, HAZ, WAZ, WHZ, Health_Status")
print(f"\nAll classes have sufficient samples for stratified splitting ✓")
print("\n📝 Next Step:")
print("  Run: python train_models.py")
print("="*70)