# Drug Accuracy Prediction Model

A machine learning project that predicts the most suitable drug for patients based on their medical characteristics using Random Forest classification.

## 🎯 Project Overview

This project implements a drug recommendation system that analyzes patient data to predict which drug would be most effective. The model achieves **97.5% accuracy** in predicting drug outcomes for patients based on their age, sex, blood pressure, cholesterol levels, and sodium-to-potassium ratio.

## 📊 Dataset

- **Source**: `data/drug200.csv`
- **Size**: 200 patient records
- **Features**: 5 medical characteristics
- **Target**: 5 different drug types
- **Quality**: No missing values, well-balanced dataset

### Feature Description
| Feature | Type | Description | Values |
|---------|------|-------------|---------|
| Age | Numeric | Patient age | 15-74 years |
| Sex | Categorical | Patient gender | M (Male), F (Female) |
| BP | Categorical | Blood pressure level | LOW, NORMAL, HIGH |
| Cholesterol | Categorical | Cholesterol level | NORMAL, HIGH |
| Na_to_K | Numeric | Sodium to Potassium ratio | 6.27-38.25 |

### Target Classes
- **DrugY**: 91 patients (45.5%)
- **drugX**: 54 patients (27.0%)
- **drugA**: 23 patients (11.5%)
- **drugC**: 16 patients (8.0%)
- **drugB**: 16 patients (8.0%)

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

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

## 📁 Project Structure

```
DrugAccuracy/
├── data/
│   └── drug200.csv                    # Dataset
├── scripts/
│   ├── 1.0-tnt-drug-prediction.py     # Main training script
│   └── 2.0-tnt-drug-predictor.py      # Interactive prediction tool
├── output/
│   ├── random_forest_drug_model.pkl   # Trained model
│   ├── label_encoders.pkl             # Categorical encoders
│   └── drug_prediction_analysis.png   # Visualizations
├── results/
│   ├── drug_prediction_results.txt    # Detailed results
│   └── model_comparison.csv           # Performance metrics
├── MODEL_SUMMARY.md                   # Technical documentation
└── readme.md                          # This file
```

## 🎯 Model Performance

### Metrics
- **Accuracy**: 97.5%
- **Precision**: 97.6%
- **Recall**: 97.5%
- **F1-Score**: 97.5%

### Feature Importance
1. **Na_to_K ratio** (54.6%) - Primary decision factor
2. **Blood Pressure** (24.8%) - Secondary factor
3. **Age** (13.6%) - Moderate influence
4. **Cholesterol** (5.5%) - Minor influence
5. **Sex** (1.6%) - Minimal influence

### Classification Report
```
              precision    recall  f1-score   support
       DrugY       0.95      1.00      0.97        18
       drugA       1.00      1.00      1.00         5
       drugB       1.00      1.00      1.00         3
       drugC       1.00      1.00      1.00         3
       drugX       1.00      0.91      0.95        11
    accuracy                           0.97        40
```

## 🔧 Usage Examples

### Command Line Prediction
```python
# Example patient data
patient = {
    'age': 25,
    'sex': 'F',
    'bp': 'HIGH',
    'cholesterol': 'HIGH',
    'na_to_k': 25.0
}
# Predicted: DrugY
```

### Batch Predictions
The model can process multiple patients simultaneously and provides confidence scores for each prediction.

## 📈 Visualizations

The model generates comprehensive visualizations:
- **Drug Distribution**: Pie chart showing class balance
- **Feature Importance**: Bar chart comparing feature contributions
- **Model Comparison**: Accuracy comparison between algorithms
- **Confusion Matrix**: Detailed prediction accuracy by class

## 🧠 Model Architecture

### Algorithm: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Random State**: 42 (for reproducibility)
- **Strategy**: Ensemble learning with majority voting
- **Advantages**: High accuracy, handles categorical data well, provides feature importance

### Data Preprocessing
1. **Label Encoding**: Categorical variables converted to numeric
2. **Train-Test Split**: 80-20 stratified split
3. **No Scaling**: Tree-based models don't require feature scaling

## 🔍 Key Insights

1. **Sodium-to-Potassium ratio is crucial** - This biochemical marker is the strongest predictor of drug effectiveness
2. **Blood pressure matters significantly** - Hypertension status heavily influences drug selection
3. **Age is moderately important** - Older patients may require different medications
4. **Gender has minimal impact** - Drug selection is largely independent of patient sex
5. **Model is highly reliable** - 97.5% accuracy indicates strong predictive power

## 🛠️ Technical Details

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

### Model Serialization
- Models saved using joblib for efficient loading
- Label encoders preserved for consistent preprocessing
- Cross-platform compatibility ensured

## 📋 Future Enhancements

### Short Term
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for robust evaluation
- [ ] Additional algorithms (XGBoost, SVM)
- [ ] Web interface for easy access

### Long Term
- [ ] Real-time prediction API
- [ ] Integration with electronic health records
- [ ] Explainable AI features (SHAP values)
- [ ] Larger dataset incorporation
- [ ] Clinical validation studies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**TNT**
- Version: 1.0
- Date: August 2025
- Contact: [Your contact information]

## 🙏 Acknowledgments

- Dataset provided for educational purposes
- Scikit-learn community for excellent ML tools
- Healthcare professionals for domain expertise

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Random Forest Algorithm](https://en.wikipedia.org/wiki/Random_forest)
- [Drug Classification in Healthcare](https://www.who.int/medicines/areas/quality_safety/safety_efficacy/trainingcourses/definitions.pdf)

---

**⚠️ Disclaimer**: This model is for educational and research purposes only. It should not be used for actual medical decisions without proper clinical validation and healthcare professional oversight.
