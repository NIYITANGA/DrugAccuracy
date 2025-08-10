"""
Drug Prediction Model - Version 1.0
Author: TNT
Description: Machine learning model to predict drug outcomes for patients
Dataset: drug200.csv - Contains patient information and corresponding drug recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime

# Create output directories if they don't exist
os.makedirs('../output', exist_ok=True)
os.makedirs('../results', exist_ok=True)

print("="*60)
print("DRUG PREDICTION MODEL - VERSION 1.0")
print("="*60)

# Load the dataset
print("\n1. Loading Dataset...")
df = pd.read_csv('../data/drug200.csv')
print(f"Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Display basic information about the dataset
print("\n2. Dataset Overview...")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget variable distribution:")
print(df['Drug'].value_counts())

# Data preprocessing
print("\n3. Data Preprocessing...")

# Create a copy for preprocessing
df_processed = df.copy()

# Encode categorical variables
label_encoders = {}

# Encode Sex
le_sex = LabelEncoder()
df_processed['Sex_encoded'] = le_sex.fit_transform(df_processed['Sex'])
label_encoders['Sex'] = le_sex

# Encode BP (Blood Pressure)
le_bp = LabelEncoder()
df_processed['BP_encoded'] = le_bp.fit_transform(df_processed['BP'])
label_encoders['BP'] = le_bp

# Encode Cholesterol
le_chol = LabelEncoder()
df_processed['Cholesterol_encoded'] = le_chol.fit_transform(df_processed['Cholesterol'])
label_encoders['Cholesterol'] = le_chol

# Encode target variable
le_drug = LabelEncoder()
df_processed['Drug_encoded'] = le_drug.fit_transform(df_processed['Drug'])
label_encoders['Drug'] = le_drug

print("Categorical variables encoded successfully!")
print(f"Sex mapping: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print(f"BP mapping: {dict(zip(le_bp.classes_, le_bp.transform(le_bp.classes_)))}")
print(f"Cholesterol mapping: {dict(zip(le_chol.classes_, le_chol.transform(le_chol.classes_)))}")
print(f"Drug mapping: {dict(zip(le_drug.classes_, le_drug.transform(le_drug.classes_)))}")

# Prepare features and target
X = df_processed[['Age', 'Sex_encoded', 'BP_encoded', 'Cholesterol_encoded', 'Na_to_K']]
y = df_processed['Drug_encoded']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data
print("\n4. Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train models
print("\n5. Training Models...")

# Random Forest Classifier
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Decision Tree Classifier
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
print("\n6. Making Predictions...")
rf_predictions = rf_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)

# Evaluate models
print("\n7. Model Evaluation...")

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Evaluate both models
rf_results = evaluate_model(y_test, rf_predictions, "Random Forest")
dt_results = evaluate_model(y_test, dt_predictions, "Decision Tree")

# Detailed classification reports
print("\n8. Detailed Classification Reports...")

print("\nRandom Forest Classification Report:")
rf_report = classification_report(y_test, rf_predictions, 
                                target_names=le_drug.classes_, 
                                output_dict=True)
print(classification_report(y_test, rf_predictions, target_names=le_drug.classes_))

print("\nDecision Tree Classification Report:")
dt_report = classification_report(y_test, dt_predictions, 
                                target_names=le_drug.classes_, 
                                output_dict=True)
print(classification_report(y_test, dt_predictions, target_names=le_drug.classes_))

# Feature importance
print("\n9. Feature Importance Analysis...")
feature_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

print("\nRandom Forest Feature Importance:")
rf_importance = rf_model.feature_importances_
for i, importance in enumerate(rf_importance):
    print(f"{feature_names[i]}: {importance:.4f}")

print("\nDecision Tree Feature Importance:")
dt_importance = dt_model.feature_importances_
for i, importance in enumerate(dt_importance):
    print(f"{feature_names[i]}: {importance:.4f}")

# Create visualizations
print("\n10. Creating Visualizations...")

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Target distribution
axes[0, 0].pie(df['Drug'].value_counts().values, 
               labels=df['Drug'].value_counts().index, 
               autopct='%1.1f%%')
axes[0, 0].set_title('Drug Distribution in Dataset')

# 2. Feature importance comparison
x_pos = np.arange(len(feature_names))
width = 0.35
axes[0, 1].bar(x_pos - width/2, rf_importance, width, label='Random Forest', alpha=0.8)
axes[0, 1].bar(x_pos + width/2, dt_importance, width, label='Decision Tree', alpha=0.8)
axes[0, 1].set_xlabel('Features')
axes[0, 1].set_ylabel('Importance')
axes[0, 1].set_title('Feature Importance Comparison')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(feature_names, rotation=45)
axes[0, 1].legend()

# 3. Model accuracy comparison
models = ['Random Forest', 'Decision Tree']
accuracies = [rf_results['accuracy'], dt_results['accuracy']]
axes[1, 0].bar(models, accuracies, color=['skyblue', 'lightcoral'])
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Model Accuracy Comparison')
axes[1, 0].set_ylim(0, 1)
for i, v in enumerate(accuracies):
    axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')

# 4. Confusion matrix for best model (Random Forest)
cm = confusion_matrix(y_test, rf_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_drug.classes_, 
            yticklabels=le_drug.classes_, 
            ax=axes[1, 1])
axes[1, 1].set_title('Random Forest Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('../output/drug_prediction_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved to '../output/drug_prediction_analysis.png'")

# Save models and results
print("\n11. Saving Models and Results...")

# Save the best model (Random Forest)
joblib.dump(rf_model, '../output/random_forest_drug_model.pkl')
joblib.dump(label_encoders, '../output/label_encoders.pkl')
print("Random Forest model saved to '../output/random_forest_drug_model.pkl'")
print("Label encoders saved to '../output/label_encoders.pkl'")

# Save results to text file
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
results_text = f"""
DRUG PREDICTION MODEL RESULTS
Generated on: {timestamp}
================================

DATASET INFORMATION:
- Total samples: {df.shape[0]}
- Features: {df.shape[1] - 1}
- Target classes: {len(df['Drug'].unique())}
- Class distribution: {dict(df['Drug'].value_counts())}

MODEL PERFORMANCE:
Random Forest:
- Accuracy: {rf_results['accuracy']:.4f}
- Precision: {rf_results['precision']:.4f}
- Recall: {rf_results['recall']:.4f}
- F1-Score: {rf_results['f1_score']:.4f}

Decision Tree:
- Accuracy: {dt_results['accuracy']:.4f}
- Precision: {dt_results['precision']:.4f}
- Recall: {dt_results['recall']:.4f}
- F1-Score: {dt_results['f1_score']:.4f}

FEATURE IMPORTANCE (Random Forest):
"""

for i, importance in enumerate(rf_importance):
    results_text += f"- {feature_names[i]}: {importance:.4f}\n"

results_text += f"""
ENCODING MAPPINGS:
- Sex: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}
- BP: {dict(zip(le_bp.classes_, le_bp.transform(le_bp.classes_)))}
- Cholesterol: {dict(zip(le_chol.classes_, le_chol.transform(le_chol.classes_)))}
- Drug: {dict(zip(le_drug.classes_, le_drug.transform(le_drug.classes_)))}

CONCLUSION:
The Random Forest model achieved the best performance with {rf_results['accuracy']:.1%} accuracy.
The most important features for drug prediction are:
1. {feature_names[np.argmax(rf_importance)]} ({max(rf_importance):.4f})
2. {feature_names[np.argsort(rf_importance)[-2]]} ({sorted(rf_importance)[-2]:.4f})
3. {feature_names[np.argsort(rf_importance)[-3]]} ({sorted(rf_importance)[-3]:.4f})
"""

with open('../results/drug_prediction_results.txt', 'w') as f:
    f.write(results_text)

print("Results saved to '../results/drug_prediction_results.txt'")

# Save detailed results as CSV
results_df = pd.DataFrame({
    'Model': ['Random Forest', 'Decision Tree'],
    'Accuracy': [rf_results['accuracy'], dt_results['accuracy']],
    'Precision': [rf_results['precision'], dt_results['precision']],
    'Recall': [rf_results['recall'], dt_results['recall']],
    'F1_Score': [rf_results['f1_score'], dt_results['f1_score']]
})

results_df.to_csv('../results/model_comparison.csv', index=False)
print("Model comparison saved to '../results/model_comparison.csv'")

# Example prediction function
def predict_drug(age, sex, bp, cholesterol, na_to_k):
    """
    Predict drug for a new patient
    
    Parameters:
    age: int - Patient age
    sex: str - 'M' or 'F'
    bp: str - 'LOW', 'NORMAL', or 'HIGH'
    cholesterol: str - 'NORMAL' or 'HIGH'
    na_to_k: float - Sodium to Potassium ratio
    
    Returns:
    str - Predicted drug
    """
    # Encode the input
    sex_encoded = label_encoders['Sex'].transform([sex])[0]
    bp_encoded = label_encoders['BP'].transform([bp])[0]
    chol_encoded = label_encoders['Cholesterol'].transform([cholesterol])[0]
    
    # Make prediction
    features = np.array([[age, sex_encoded, bp_encoded, chol_encoded, na_to_k]])
    prediction_encoded = rf_model.predict(features)[0]
    
    # Decode prediction
    predicted_drug = label_encoders['Drug'].inverse_transform([prediction_encoded])[0]
    
    return predicted_drug

# Test the prediction function
print("\n12. Testing Prediction Function...")
test_cases = [
    (25, 'F', 'HIGH', 'HIGH', 25.0),
    (50, 'M', 'LOW', 'NORMAL', 15.0),
    (35, 'F', 'NORMAL', 'HIGH', 10.0)
]

print("\nSample Predictions:")
for i, (age, sex, bp, chol, na_k) in enumerate(test_cases, 1):
    predicted = predict_drug(age, sex, bp, chol, na_k)
    print(f"Patient {i}: Age={age}, Sex={sex}, BP={bp}, Cholesterol={chol}, Na_to_K={na_k:.1f}")
    print(f"Predicted Drug: {predicted}")
    print()

print("="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Best Model: Random Forest (Accuracy: {rf_results['accuracy']:.1%})")
print("Files saved:")
print("- Model: ../output/random_forest_drug_model.pkl")
print("- Encoders: ../output/label_encoders.pkl")
print("- Visualization: ../output/drug_prediction_analysis.png")
print("- Results: ../results/drug_prediction_results.txt")
print("- Comparison: ../results/model_comparison.csv")
print("="*60)
