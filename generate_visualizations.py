"""
IRERERO PROJECT - Visualization Generation Script
This script generates all 6 visualizations needed for the research report.

Author: IRERERO Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os

print("="*70)
print("IRERERO VISUALIZATION GENERATION")
print("Generating figures for research report")
print("="*70)

# ============================================
# CHECK REQUIRED FILES
# ============================================
print("\n[0/6] Checking required files...")

required_files = [
    'training_results.csv',
    'validation_results.csv',
    'test_results.csv',
    'irerero_model_training_data.csv',
    'best_health_model.pkl',
    'feature_scaler.pkl'
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print("\n❌ ERROR: Missing required files:")
    for f in missing_files:
        print(f"   - {f}")
    print("\n📝 Please run these scripts first:")
    print("   1. python generate_dataset.py")
    print("   2. python train_models.py")
    exit(1)

print("✓ All required files found")

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Load results
train_df = pd.read_csv('training_results.csv', index_col=0)
val_df = pd.read_csv('validation_results.csv', index_col=0)
test_df = pd.read_csv('test_results.csv', index_col=0)

# Load dataset and model
df = pd.read_csv('irerero_model_training_data.csv')
best_model = joblib.load('best_health_model.pkl')

# Prepare data
X = df[['Age_Months', 'Gender_Numeric', 'Height_cm', 'Weight_kg', 
        'BMI', 'HAZ', 'WAZ', 'WHZ']]
y = df['Health_Status']

# ============================================
# VISUALIZATION 1: Model Accuracy Comparison
# ============================================
print("\n[1/6] Generating Model Accuracy Comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(test_df.index))
width = 0.25

bars1 = ax.bar(x - width, train_df['Accuracy'], width, label='Training', 
               color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x, val_df['Accuracy'], width, label='Validation', 
               color='#3498db', alpha=0.8)
bars3 = ax.bar(x + width, test_df['Accuracy'], width, label='Testing', 
               color='#e74c3c', alpha=0.8)

ax.set_xlabel('Machine Learning Models', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy Score', fontsize=13, fontweight='bold')
ax.set_title('Model Performance Comparison Across Training, Validation, and Testing Datasets', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(test_df.index, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.1])

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('Fig1_Model_Accuracy_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Fig1_Model_Accuracy_Comparison.png")
print("📸 INSERT THIS IN REPORT: Section 5.3 (Model Accuracy Comparison)")
plt.close()

# ============================================
# VISUALIZATION 2: Confusion Matrix
# ============================================
print("\n[2/6] Generating Confusion Matrix for Best Model...")

# Get test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=42, stratify=y)

# Check if model needs scaling
model_name = type(best_model).__name__
if 'KNeighbors' in model_name or 'SVC' in model_name or 'LogisticRegression' in model_name:
    try:
        scaler = joblib.load('feature_scaler.pkl')
        X_test_processed = scaler.transform(X_test)
    except FileNotFoundError:
        X_test_processed = X_test.values
else:
    X_test_processed = X_test.values

# Get predictions
y_pred = best_model.predict(X_test_processed)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
classes = sorted(y.unique())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'}, linewidths=0.5)
plt.title(f'Confusion Matrix - {type(best_model).__name__} (Best Model)', 
          fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Actual Health Status', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Health Status', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Fig2_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Fig2_Confusion_Matrix.png")
print("📸 INSERT THIS IN REPORT: Section 5.4 (Confusion Matrix Analysis)")
plt.close()

# ============================================
# VISUALIZATION 3: Feature Importance
# ============================================
print("\n[3/6] Generating Feature Importance Plot...")

# Get feature importance (works for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    features = ['Age (months)', 'Gender', 'Height (cm)', 'Weight (kg)', 
                'BMI', 'HAZ', 'WAZ', 'WHZ']
    
    # Create DataFrame
    feat_imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], 
             color='#9b59b6', alpha=0.8)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Feature Importance Analysis for Child Health Prediction', 
              fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(feat_imp_df['Importance']):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('Fig3_Feature_Importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig3_Feature_Importance.png")
    print("📸 INSERT THIS IN REPORT: Section 5.5 (Feature Importance)")
    plt.close()
else:
    print("⚠ Feature importance not available for this model type")
    print("  Creating correlation heatmap instead...")
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[['Age_Months', 'Gender_Numeric', 'Height_cm', 'Weight_kg', 
                      'BMI', 'HAZ', 'WAZ', 'WHZ']].corr()
    
    labels = ['Age', 'Gender', 'Height', 'Weight', 'BMI', 'HAZ', 'WAZ', 'WHZ']
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1,
                xticklabels=labels, yticklabels=labels)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('Fig3_Feature_Importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: Fig3_Feature_Importance.png (Correlation Matrix)")
    print("📸 INSERT THIS IN REPORT: Section 5.5 (Feature Analysis)")
    plt.close()

# ============================================
# VISUALIZATION 4: Performance Metrics Comparison
# ============================================
print("\n[4/6] Generating Performance Metrics Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
    values = test_df[metric].values
    models = test_df.index
    
    bars = ax.barh(models, values, color=color, alpha=0.8)
    ax.set_xlabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1.1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=10, fontweight='bold')

plt.suptitle('Comprehensive Performance Metrics for All Models', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('Fig4_Performance_Metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Fig4_Performance_Metrics.png")
print("📸 INSERT THIS IN REPORT: Section 5.6 (Performance Metrics)")
plt.close()

# ============================================
# VISUALIZATION 5: Health Status Distribution
# ============================================
print("\n[5/6] Generating Health Status Distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Bar chart
status_counts = df['Health_Status'].value_counts()
colors_status = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#3498db']
bars = ax1.bar(status_counts.index, status_counts.values, 
               color=colors_status[:len(status_counts)], alpha=0.8)
ax1.set_xlabel('Health Status', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Children', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Child Health Status in Dataset', 
              fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Pie chart
ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
        colors=colors_status[:len(status_counts)], startangle=90, 
        textprops={'fontsize': 11})
ax2.set_title('Percentage Distribution of Health Status', 
              fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('Fig5_Health_Status_Distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Fig5_Health_Status_Distribution.png")
print("📸 INSERT THIS IN REPORT: Section 4.3 (Dataset Description)")
plt.close()

# ============================================
# VISUALIZATION 6: Age vs BMI by Health Status
# ============================================
print("\n[6/6] Generating Age vs BMI Scatter Plot...")

plt.figure(figsize=(14, 8))

for status in sorted(df['Health_Status'].unique()):
    subset = df[df['Health_Status'] == status]
    plt.scatter(subset['Age_Months'], subset['BMI'], 
                label=status, alpha=0.6, s=50)

plt.xlabel('Age (Months)', fontsize=12, fontweight='bold')
plt.ylabel('BMI (Body Mass Index)', fontsize=12, fontweight='bold')
plt.title('Relationship Between Age and BMI Across Different Health Statuses', 
          fontsize=14, fontweight='bold', pad=15)
plt.legend(title='Health Status', fontsize=10, title_fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Fig6_Age_BMI_Scatter.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Fig6_Age_BMI_Scatter.png")
print("📸 INSERT THIS IN REPORT: Section 5.7 (Data Distribution Analysis)")
plt.close()

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Figures (Ready for Report):")
print("  1. Fig1_Model_Accuracy_Comparison.png → Section 5.3")
print("  2. Fig2_Confusion_Matrix.png → Section 5.4")
print("  3. Fig3_Feature_Importance.png → Section 5.5")
print("  4. Fig4_Performance_Metrics.png → Section 5.6")
print("  5. Fig5_Health_Status_Distribution.png → Section 4.3")
print("  6. Fig6_Age_BMI_Scatter.png → Section 5.7")
print("\n📝 Next Steps:")
print("  1. Open your Word document")
print("  2. Insert each figure in the section indicated above")
print("  3. Add captions below each figure")
print("  4. Update table of figures")
print("="*70)