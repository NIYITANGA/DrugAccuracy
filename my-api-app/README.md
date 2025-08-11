---
title: Drug Prediction System
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🏥 Drug Prediction System

An AI-powered drug recommendation system that predicts the most suitable drug for patients based on their medical characteristics using Random Forest classification.

## 🎯 Overview

This application uses machine learning to analyze patient data and recommend the most appropriate drug treatment. The model achieves **97.5% accuracy** in predicting drug outcomes based on:

- **Age**: Patient age (0-120 years)
- **Sex**: M (Male) or F (Female) 
- **Blood Pressure**: LOW, NORMAL, or HIGH
- **Cholesterol**: NORMAL or HIGH
- **Na_to_K Ratio**: Sodium to Potassium ratio (0-50)

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 97.5%
- **Training Data**: 200 patient records
- **Target Classes**: DrugY, drugX, drugA, drugB, drugC

### Feature Importance
1. **Na_to_K Ratio** (54.6%) - Primary decision factor
2. **Blood Pressure** (24.8%) - Secondary factor
3. **Age** (13.6%) - Moderate influence
4. **Cholesterol** (5.5%) - Minor influence
5. **Sex** (1.6%) - Minimal influence

## 🚀 How to Use

1. **Enter Patient Information**: Fill in the patient's medical characteristics
2. **Click Predict**: Get instant drug recommendation with confidence score
3. **Review Results**: See detailed analysis and probability breakdown
4. **Try Examples**: Use pre-loaded example patients to test the system

## 📊 Example Predictions

The app includes several example patients that demonstrate different prediction scenarios:

- **High Na_to_K + High BP/Cholesterol** → Usually predicts DrugY
- **Low BP conditions** → Often predicts drugC
- **Normal conditions with moderate Na_to_K** → Typically predicts drugX
- **High BP + High Cholesterol (lower Na_to_K)** → May predict drugA
- **Older patients with specific conditions** → Could predict drugB

## 🔬 Technical Implementation

The app uses:
- **Gradio** for the interactive web interface
- **Scikit-learn** for machine learning
- **Pandas/NumPy** for data processing
- **Random Forest** algorithm for predictions

## ⚠️ Medical Disclaimer

**IMPORTANT**: This AI model is for educational and research purposes only. The predictions should not be used for actual medical decisions without proper clinical validation and healthcare professional oversight. Always consult with qualified healthcare providers for medical advice.

## 🛠️ Development

### Local Setup
```bash
pip install -r requirements.txt
python app.py
```

### Model Training
The model was trained on a dataset of 200 patient records with the following distribution:
- DrugY: 45.5% of cases
- drugX: 27.0% of cases  
- drugA: 11.5% of cases
- drugC: 8.0% of cases
- drugB: 8.0% of cases

## 📈 Performance Metrics

- **Accuracy**: 97.5%
- **Precision**: 97.6%
- **Recall**: 97.5%
- **F1-Score**: 97.5%

## 👨‍💻 Author

**TNT** - Machine Learning Engineer
- Version: 3.0
- Date: August 2025

## 📄 License

This project is licensed under the MIT License.

---

**Built with ❤️ using Gradio and Hugging Face Spaces**
