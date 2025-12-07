"""
IRERERO PROJECT - Model Training Script
This script trains 6 ML models and generates CSV results for the report.

Author: IRERERO Team
Date: December 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IRERERO MODEL TRAINING PIPELINE")
print("Training 6 Machine Learning Models")
print("="*70)

# ============================================
# STEP 1: Load Dataset
# ============================================
print("\n[1/5] Loading dataset...")
try:
    df = pd.read_csv('irerero_model_training_data.csv')
    print(f"✓ Dataset loaded successfully")
    print(f"  - Total samples: {df.shape[0]}")
    print(f"  - Total features: {df.shape[1]}")
except FileNotFoundError:
    print("❌ ERROR: 'irerero_model_training_data.csv' not found!")
    print("   Please run 'python generate_dataset.py' first.")
    exit(1)

# Prepare features and target
X = df[['Age_Months', 'Gender_Numeric', 'Height_cm', 'Weight_kg', 
        'BMI', 'HAZ', 'WAZ', 'WHZ']]
y = df['Health_Status']

print(f"\n✓ Features: {list(X.columns)}")
print(f"✓ Target classes: {list(y.unique())}")
print(f"✓ Class distribution:")
class_counts = y.value_counts()
for cls, count in class_counts.items():
    print(f"  - {cls}: {count}")

# Check if all classes have sufficient samples for stratification
min_class_size = class_counts.min()
if min_class_size < 3:
    print(f"\n⚠️  WARNING: Smallest class has only {min_class_size} samples")
    print("   Regenerating dataset with balanced classes...")
    print("   Please run: python generate_dataset.py")
    exit(1)

# ============================================
# STEP 2: Split Data (70% / 15% / 15%)
# ============================================
print("\n[2/5] Splitting data...")

try:
    # First split: 85% temp, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Second split: 70% train, 15% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"✓ Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    
except ValueError as e:
    print(f"\n❌ ERROR during data splitting: {e}")
    print("\n💡 SOLUTION: Re-run dataset generation to ensure balanced classes")
    print("   Command: python generate_dataset.py")
    exit(1)

# ============================================
# STEP 3: Define Models (EXACTLY 6 models)
# ============================================
print("\n[3/5] Initializing 6 machine learning models...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        max_depth=10
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=5
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42,
        multi_class='multinomial'
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5
    ),
    'Support Vector Machine': SVC(
        kernel='rbf', 
        random_state=42
    )
}

print(f"✓ Models initialized:")
for i, name in enumerate(models.keys(), 1):
    print(f"  {i}. {name}")

# ============================================
# STEP 4: Prepare Scaled Data
# ============================================
print("\n[4/5] Preparing data for training...")

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'feature_scaler.pkl')
print("✓ Feature scaler saved: feature_scaler.pkl")

# ============================================
# STEP 5: Train and Evaluate All Models
# ============================================
print("\n[5/5] Training and evaluating models...")
print("="*70)

# Storage for results
results = {
    'training': [],
    'validation': [],
    'test': []
}

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Determine if model needs scaling
    needs_scaling = name in ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine']
    
    if needs_scaling:
        X_train_use = X_train_scaled
        X_val_use = X_val_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train.values
        X_val_use = X_val.values
        X_test_use = X_test.values
    
    # Train the model
    model.fit(X_train_use, y_train)
    
    # Evaluate on all three datasets
    datasets = [
        ('training', X_train_use, y_train),
        ('validation', X_val_use, y_val),
        ('test', X_test_use, y_test)
    ]
    
    for dataset_name, X_data, y_data in datasets:
        # Make predictions
        y_pred = model.predict(X_data)
        
        # Calculate metrics
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_data, y_pred),
            'Precision': precision_score(y_data, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_data, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_data, y_pred, average='weighted', zero_division=0)
        }
        
        results[dataset_name].append(metrics)
        
        # Print metrics
        if dataset_name == 'training':
            print(f"  ✓ Training Accuracy: {metrics['Accuracy']:.4f}")
        elif dataset_name == 'validation':
            print(f"  ✓ Validation Accuracy: {metrics['Accuracy']:.4f}")
        else:
            print(f"  ✓ Test Accuracy: {metrics['Accuracy']:.4f}")
    
    # Track best model (based on validation accuracy)
    val_accuracy = results['validation'][-1]['Accuracy']
    if val_accuracy > best_score:
        best_score = val_accuracy
        best_model = model
        best_model_name = name

# Save best model
joblib.dump(best_model, 'best_health_model.pkl')
print(f"\n{'='*70}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Validation Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
print(f"✓ Saved: best_health_model.pkl")
print(f"{'='*70}")

# ============================================
# STEP 6: Save Results to CSV Files
# ============================================
print("\n📊 Saving results to CSV files...")

# Create DataFrames with Model as index
for dataset_name, data in results.items():
    df_results = pd.DataFrame(data)
    df_results.set_index('Model', inplace=True)
    
    filename = f'{dataset_name}_results.csv'
    df_results.to_csv(filename)
    print(f"✓ Saved: {filename}")

# Display final results table
print("\n" + "="*70)
print("TEST SET RESULTS (for report Table 3):")
print("="*70)
test_results_df = pd.DataFrame(results['test']).set_index('Model')
print(test_results_df.to_string())

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files:")
print("  1. training_results.csv    → Use for Report Table 1")
print("  2. validation_results.csv  → Use for Report Table 2")
print("  3. test_results.csv        → Use for Report Table 3")
print("  4. best_health_model.pkl   → Trained model")
print("  5. feature_scaler.pkl      → For data preprocessing")
print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
print("\n📝 Next Step:")
print("  Run: python generate_visualizations.py")
print("  This will create all 6 figures for your report")
print("="*70)