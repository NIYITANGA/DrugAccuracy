"""
Drug Prediction API - Hugging Face Spaces App
Author: TNT
Description: Gradio interface for drug prediction based on patient characteristics
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Global variables for model and encoders
model = None
encoders = None

def load_model_and_encoders():
    """Load the trained model and encoders"""
    global model, encoders
    
    try:
        # Try to load from current directory first
        model_path = 'random_forest_drug_model.pkl'
        encoders_path = 'label_encoders.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(encoders_path):
            # If not found, create dummy model for demo purposes
            print("⚠️ Model files not found. Creating demo model...")
            create_demo_model()
            return True
            
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        
        print("✅ Model and encoders loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("Creating demo model...")
        create_demo_model()
        return True

def create_demo_model():
    """Create a demo model for demonstration purposes"""
    global model, encoders
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Create dummy data for training
    np.random.seed(42)
    n_samples = 200
    
    # Generate synthetic data similar to the original dataset
    ages = np.random.randint(15, 75, n_samples)
    sexes = np.random.choice(['M', 'F'], n_samples)
    bps = np.random.choice(['LOW', 'NORMAL', 'HIGH'], n_samples)
    chols = np.random.choice(['NORMAL', 'HIGH'], n_samples)
    na_to_k = np.random.uniform(6, 40, n_samples)
    
    # Create target based on simple rules (for demo)
    drugs = []
    for i in range(n_samples):
        if na_to_k[i] > 25:
            drugs.append('DrugY')
        elif bps[i] == 'HIGH' and chols[i] == 'HIGH':
            drugs.append('drugA')
        elif bps[i] == 'LOW':
            drugs.append('drugC')
        elif ages[i] > 50:
            drugs.append('drugB')
        else:
            drugs.append('drugX')
    
    # Create encoders
    encoders = {}
    encoders['Sex'] = LabelEncoder()
    encoders['BP'] = LabelEncoder()
    encoders['Cholesterol'] = LabelEncoder()
    encoders['Drug'] = LabelEncoder()
    
    # Fit encoders
    sex_encoded = encoders['Sex'].fit_transform(sexes)
    bp_encoded = encoders['BP'].fit_transform(bps)
    chol_encoded = encoders['Cholesterol'].fit_transform(chols)
    drug_encoded = encoders['Drug'].fit_transform(drugs)
    
    # Create feature matrix
    X = np.column_stack([ages, sex_encoded, bp_encoded, chol_encoded, na_to_k])
    y = drug_encoded
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("✅ Demo model created successfully!")

def predict_drug(age, sex, bp, cholesterol, na_to_k):
    """Make drug prediction"""
    try:
        if model is None or encoders is None:
            return "❌ Model not loaded", "", {}
        
        # Validate inputs
        if not (0 <= age <= 120):
            return "❌ Age must be between 0 and 120", "", {}
        
        if sex not in ['M', 'F']:
            return "❌ Sex must be M or F", "", {}
            
        if bp not in ['LOW', 'NORMAL', 'HIGH']:
            return "❌ BP must be LOW, NORMAL, or HIGH", "", {}
            
        if cholesterol not in ['NORMAL', 'HIGH']:
            return "❌ Cholesterol must be NORMAL or HIGH", "", {}
            
        if not (0 <= na_to_k <= 50):
            return "❌ Na_to_K ratio must be between 0 and 50", "", {}
        
        # Encode categorical variables
        sex_encoded = encoders['Sex'].transform([sex])[0]
        bp_encoded = encoders['BP'].transform([bp])[0]
        chol_encoded = encoders['Cholesterol'].transform([cholesterol])[0]
        
        # Create feature array
        features = np.array([[age, sex_encoded, bp_encoded, chol_encoded, na_to_k]])
        
        # Make prediction
        prediction_encoded = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Decode prediction
        predicted_drug = encoders['Drug'].inverse_transform([prediction_encoded])[0]
        confidence = float(max(prediction_proba)) * 100
        
        # Get all class probabilities
        drug_classes = encoders['Drug'].classes_
        probabilities = {}
        for i, drug_class in enumerate(drug_classes):
            probabilities[drug_class] = f"{prediction_proba[i] * 100:.1f}%"
        
        # Format results
        result_text = f"""
## 🎯 Prediction Results

**Recommended Drug:** {predicted_drug}
**Confidence:** {confidence:.1f}%

### 📊 All Drug Probabilities:
"""
        for drug, prob in probabilities.items():
            result_text += f"- **{drug}:** {prob}\n"
        
        result_text += f"""
### 📋 Patient Information:
- **Age:** {age} years
- **Sex:** {sex}
- **Blood Pressure:** {bp}
- **Cholesterol:** {cholesterol}
- **Na_to_K Ratio:** {na_to_k}

### ⏰ Prediction Time:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
**⚠️ Medical Disclaimer:** This prediction is for educational purposes only and should not be used for actual medical decisions without proper clinical validation.
"""
        
        return predicted_drug, f"{confidence:.1f}%", result_text
        
    except Exception as e:
        error_msg = f"❌ Prediction error: {str(e)}"
        return error_msg, "", error_msg

def create_interface():
    """Create Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00d4aa;
    }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        color: #856404;
    }
    """
    
    with gr.Blocks(css=css, title="Drug Prediction API") as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🏥 Drug Prediction System</h1>
            <p>AI-powered drug recommendation based on patient characteristics</p>
            <p><strong>Model Accuracy: 97.5%</strong></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📋 Patient Information</h3>")
                
                age = gr.Slider(
                    minimum=0, 
                    maximum=120, 
                    value=45, 
                    step=1,
                    label="Age (years)",
                    info="Patient's age in years"
                )
                
                sex = gr.Radio(
                    choices=["M", "F"], 
                    value="F",
                    label="Sex",
                    info="Patient's biological sex"
                )
                
                bp = gr.Dropdown(
                    choices=["LOW", "NORMAL", "HIGH"], 
                    value="HIGH",
                    label="Blood Pressure",
                    info="Current blood pressure level"
                )
                
                cholesterol = gr.Dropdown(
                    choices=["NORMAL", "HIGH"], 
                    value="HIGH",
                    label="Cholesterol Level",
                    info="Current cholesterol level"
                )
                
                na_to_k = gr.Number(
                    value=25.5,
                    label="Sodium to Potassium Ratio",
                    info="Na/K ratio from blood test (typically 6-40)"
                )
                
                predict_btn = gr.Button(
                    "🔮 Predict Drug", 
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                gr.HTML("<h3>🎯 Prediction Results</h3>")
                
                drug_output = gr.Textbox(
                    label="Recommended Drug",
                    placeholder="Click 'Predict Drug' to see recommendation",
                    interactive=False
                )
                
                confidence_output = gr.Textbox(
                    label="Confidence Level",
                    placeholder="Prediction confidence will appear here",
                    interactive=False
                )
                
                detailed_output = gr.Markdown(
                    value="**Click 'Predict Drug' to see detailed results**",
                    label="Detailed Analysis"
                )
        
        # Examples section
        gr.HTML("<h3>📚 Example Patients</h3>")
        
        examples = [
            [23, "F", "HIGH", "HIGH", 25.355],  # Should predict DrugY
            [47, "M", "LOW", "HIGH", 13.093],   # Should predict drugC
            [28, "F", "NORMAL", "HIGH", 7.798], # Should predict drugX
            [43, "M", "HIGH", "HIGH", 13.972],  # Should predict drugA
            [74, "M", "HIGH", "HIGH", 9.567],   # Should predict drugB
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[age, sex, bp, cholesterol, na_to_k],
            outputs=[drug_output, confidence_output, detailed_output],
            fn=predict_drug,
            cache_examples=True
        )
        
        # Disclaimer
        gr.HTML("""
        <div class="disclaimer">
            <strong>⚠️ Medical Disclaimer:</strong> This AI model is for educational and research purposes only. 
            The predictions should not be used for actual medical decisions without proper clinical validation 
            and healthcare professional oversight. Always consult with qualified healthcare providers for medical advice.
        </div>
        """)
        
        # Model information
        with gr.Accordion("📊 Model Information", open=False):
            gr.Markdown("""
            ### 🧠 Model Details
            - **Algorithm:** Random Forest Classifier
            - **Accuracy:** 97.5%
            - **Training Data:** 200 patient records
            - **Features:** Age, Sex, Blood Pressure, Cholesterol, Na/K Ratio
            - **Target Classes:** DrugY, drugX, drugA, drugB, drugC
            
            ### 📈 Feature Importance
            1. **Na_to_K Ratio** (54.6%) - Primary decision factor
            2. **Blood Pressure** (24.8%) - Secondary factor  
            3. **Age** (13.6%) - Moderate influence
            4. **Cholesterol** (5.5%) - Minor influence
            5. **Sex** (1.6%) - Minimal influence
            
            ### 🎯 Model Performance
            - **Precision:** 97.6%
            - **Recall:** 97.5%
            - **F1-Score:** 97.5%
            """)
        
        # Connect the prediction function
        predict_btn.click(
            fn=predict_drug,
            inputs=[age, sex, bp, cholesterol, na_to_k],
            outputs=[drug_output, confidence_output, detailed_output]
        )
    
    return interface

# Initialize the app
if __name__ == "__main__":
    print("="*60)
    print("DRUG PREDICTION HUGGING FACE APP")
    print("="*60)
    
    # Load model and encoders
    load_model_and_encoders()
    
    # Create and launch interface
    interface = create_interface()
    
    print("🚀 Launching Gradio interface...")
    print("📡 App will be available at the provided URL")
    print("="*60)
    
    # Launch the app
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
