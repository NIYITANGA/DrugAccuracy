# Drug Prediction Model - Project Summary

## Overview
This project implements a machine learning model to predict drug outcomes for patients based on their medical characteristics. The model uses patient information including age, sex, blood pressure, cholesterol levels, and sodium-to-potassium ratio to recommend the most suitable drug.

## Dataset
- **File**: `data/drug200.csv`
- **Samples**: 200 patients
- **Features**: 5 (Age, Sex, BP, Cholesterol, Na_to_K)
- **Target**: Drug classification (DrugY, drugX, drugA, drugB, drugC)
- **Class Distribution**:
  - DrugY: 91 patients (45.5%)
  - drugX: 54 patients (27.0%)
  - drugA: 23 patients (11.5%)
  - drugC: 16 patients (8.0%)
  - drugB: 16 patients (8.0%)

## Model Performance
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 97.5%
- **Precision**: 97.6%
- **Recall**: 97.5%
- **F1-Score**: 97.5%

## Feature Importance
1. **Na_to_K ratio** (54.6%) - Most important feature
2. **Blood Pressure** (24.8%) - Second most important
3. **Age** (13.6%) - Third most important
4. **Cholesterol** (5.5%) - Minor importance
5. **Sex** (1.6%) - Least important

## Files Structure

### Scripts
- `scripts/1.0-tnt-drug-prediction.py` - Main training script
- `scripts/2.0-tnt-drug-predictor.py` - Interactive prediction script

### Output Files
- `output/random_forest_drug_model.pkl` - Trained model
- `output/label_encoders.pkl` - Categorical encoders
- `output/drug_prediction_analysis.png` - Visualization plots

### Results
- `results/drug_prediction_results.txt` - Detailed results report
- `results/model_comparison.csv` - Model performance comparison

## How to Use

### 1. Train the Model
```bash
cd scripts
python3 1.0-tnt-drug-prediction.py
```

### 2. Make Predictions
```bash
cd scripts
python3 2.0-tnt-drug-predictor.py
```

## Model Features

### Input Features
- **Age**: Patient age (integer)
- **Sex**: M (Male) or F (Female)
- **BP**: Blood pressure (LOW, NORMAL, HIGH)
- **Cholesterol**: Cholesterol level (NORMAL, HIGH)
- **Na_to_K**: Sodium to Potassium ratio (float)

### Output
- **Drug**: Predicted drug type (DrugY, drugX, drugA, drugB, drugC)
- **Confidence**: Prediction confidence percentage

## Key Insights

1. **Na_to_K ratio is the most critical factor** in drug selection, accounting for over 54% of the decision-making process.

2. **Blood pressure is the second most important factor**, contributing nearly 25% to the prediction.

3. **Age plays a moderate role** in drug selection (13.6% importance).

4. **Sex has minimal impact** on drug selection in this dataset (1.6% importance).

5. **The model achieves excellent performance** with 97.5% accuracy, indicating strong predictive capability.

## Technical Details

### Data Preprocessing
- Categorical variables encoded using LabelEncoder
- No missing values in the dataset
- Stratified train-test split (80-20)

### Model Configuration
- Random Forest with 100 estimators
- Random state set to 42 for reproducibility
- Default hyperparameters used

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix analysis
- Feature importance analysis
- Classification report for each drug class

## Visualization
The model generates comprehensive visualizations including:
- Drug distribution pie chart
- Feature importance comparison
- Model accuracy comparison
- Confusion matrix heatmap

## Future Improvements
1. Hyperparameter tuning for better performance
2. Cross-validation for more robust evaluation
3. Additional algorithms comparison (SVM, XGBoost, etc.)
4. Feature engineering for better predictions
5. Model interpretability analysis (SHAP values)

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

---
**Author**: TNT  
**Version**: 1.0  
**Date**: August 2025
