"""
Drug Predictor - Version 2.0
Author: TNT
Description: Script to use the trained model for predicting drugs for new patients
"""

import joblib
import numpy as np
import pandas as pd

def load_model():
    """Load the trained model and encoders"""
    try:
        model = joblib.load('../output/random_forest_drug_model.pkl')
        encoders = joblib.load('../output/label_encoders.pkl')
        print("Model and encoders loaded successfully!")
        return model, encoders
    except FileNotFoundError:
        print("Error: Model files not found. Please run 1.0-tnt-drug-prediction.py first.")
        return None, None

def predict_drug(model, encoders, age, sex, bp, cholesterol, na_to_k):
    """
    Predict drug for a new patient
    
    Parameters:
    model: Trained RandomForest model
    encoders: Dictionary of label encoders
    age: int - Patient age
    sex: str - 'M' or 'F'
    bp: str - 'LOW', 'NORMAL', or 'HIGH'
    cholesterol: str - 'NORMAL' or 'HIGH'
    na_to_k: float - Sodium to Potassium ratio
    
    Returns:
    str - Predicted drug
    float - Prediction confidence (max probability)
    """
    try:
        # Encode the input
        sex_encoded = encoders['Sex'].transform([sex])[0]
        bp_encoded = encoders['BP'].transform([bp])[0]
        chol_encoded = encoders['Cholesterol'].transform([cholesterol])[0]
        
        # Make prediction
        features = np.array([[age, sex_encoded, bp_encoded, chol_encoded, na_to_k]])
        prediction_encoded = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Decode prediction
        predicted_drug = encoders['Drug'].inverse_transform([prediction_encoded])[0]
        confidence = max(prediction_proba)
        
        return predicted_drug, confidence
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def main():
    print("="*60)
    print("DRUG PREDICTOR - VERSION 2.0")
    print("="*60)
    
    # Load model
    model, encoders = load_model()
    if model is None:
        return
    
    print("\nModel Information:")
    print("- Algorithm: Random Forest")
    print("- Accuracy: 97.5%")
    print("- Features: Age, Sex, BP, Cholesterol, Na_to_K")
    print("- Target: Drug (DrugY, drugX, drugA, drugB, drugC)")
    
    # Interactive prediction
    print("\n" + "="*60)
    print("INTERACTIVE DRUG PREDICTION")
    print("="*60)
    
    while True:
        try:
            print("\nEnter patient information:")
            age = int(input("Age: "))
            sex = input("Sex (M/F): ").upper()
            bp = input("Blood Pressure (LOW/NORMAL/HIGH): ").upper()
            cholesterol = input("Cholesterol (NORMAL/HIGH): ").upper()
            na_to_k = float(input("Na_to_K ratio: "))
            
            # Validate inputs
            if sex not in ['M', 'F']:
                print("Error: Sex must be 'M' or 'F'")
                continue
            if bp not in ['LOW', 'NORMAL', 'HIGH']:
                print("Error: BP must be 'LOW', 'NORMAL', or 'HIGH'")
                continue
            if cholesterol not in ['NORMAL', 'HIGH']:
                print("Error: Cholesterol must be 'NORMAL' or 'HIGH'")
                continue
            
            # Make prediction
            predicted_drug, confidence = predict_drug(model, encoders, age, sex, bp, cholesterol, na_to_k)
            
            if predicted_drug:
                print(f"\n{'='*40}")
                print("PREDICTION RESULT")
                print(f"{'='*40}")
                print(f"Recommended Drug: {predicted_drug}")
                print(f"Confidence: {confidence:.2%}")
                print(f"{'='*40}")
            
            # Ask if user wants to continue
            continue_pred = input("\nMake another prediction? (y/n): ").lower()
            if continue_pred != 'y':
                break
                
        except ValueError:
            print("Error: Please enter valid numeric values for age and Na_to_K ratio.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nThank you for using the Drug Predictor!")

if __name__ == "__main__":
    main()
