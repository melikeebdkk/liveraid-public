"""
HCC Risk Assessment Model - Final Version
Exactly replicating the training preprocessing from nihai_hcc.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import joblib
import os
from sklearn.preprocessing import StandardScaler


class HCCRiskModelFinal:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'svm_best_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'hcc_scaler.pkl')
        
        self.model = None
        self.scaler = None
        
        # These are the exact columns from the training script
        self.categorical_cols = ['Gender', 'Obesity']
        self.all_feature_names = [
            'Age', 'Gender', 'AST', 'ALT', 'Albumin', 'Creatinine', 'INR', 
            'Trombosit', 'Total_Bil', 'Dir_Bil', 'Obesity', 'ALP', 'AFP'
        ]
        # Numerical columns are all except categorical
        self.numerical_cols = [col for col in self.all_feature_names if col not in self.categorical_cols]
        
        # Form field mapping to dataset column names
        self.field_mapping = {
            'age': 'Age',
            'gender': 'Gender',
            'ast': 'AST',
            'alt': 'ALT',
            'albumin': 'Albumin',
            'creatinine': 'Creatinine',
            'inr': 'INR',
            'trombosit': 'Trombosit',
            'total_bilirubin': 'Total_Bil',
            'direct_bilirubin': 'Dir_Bil',
            'obesity': 'Obesity',
            'alp': 'ALP',
            'afp': 'AFP'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler exactly as used in training"""
        try:
            # Load the trained SVM model
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ HCC SVM model loaded successfully")
            else:
                print(f"❌ Model file not found: {self.model_path}")
                return
                
            # Load the exact scaler used during training
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ HCC StandardScaler loaded successfully")
            else:
                print(f"❌ Scaler file not found: {self.scaler_path}")
                return
                
            print("✅ HCC model ready for predictions")
            
        except Exception as e:
            print(f"❌ Error loading HCC model: {e}")
            self.model = None
            self.scaler = None
    
    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict HCC risk using the exact same preprocessing as training
        """
        try:
            if self.model is None or self.scaler is None:
                raise Exception("Model or scaler not loaded properly")
            
            # Map form fields to dataset column names and prepare features
            features = {}
            missing_fields = []
            
            for form_field, dataset_col in self.field_mapping.items():
                if form_field in patient_data:
                    value = patient_data[form_field]
                    # Special handling for Trombosit - multiply by 1000 for HCC model
                    if dataset_col == 'Trombosit':
                        value = float(value) * 1000
                    features[dataset_col] = value
                else:
                    # Only allow certain fields to be missing with defaults
                    if dataset_col == 'AFP':
                        # AFP is optional, default to 0 if not provided
                        features[dataset_col] = 0.0
                    elif dataset_col == 'Obesity':
                        # Obesity is optional, default to 0 (not obese) if not provided
                        features[dataset_col] = 0
                    else:
                        # All other fields are required
                        missing_fields.append(form_field)
            
            print("📊 HCC Model - Mapped feature data:")
            for dataset_col in self.all_feature_names:
                if dataset_col in features:
                    value = features[dataset_col]
                    print(f"  {dataset_col}: {value}")
                else:
                    print(f"  {dataset_col}: MISSING")
            
            # Raise error if any required fields are missing
            if missing_fields:
                raise ValueError(f"Missing required fields for HCC prediction: {', '.join(missing_fields)}")
            
            # Create DataFrame with single row (exactly like training)
            df = pd.DataFrame([features])
            
            # Ensure all columns are present and in correct order
            for col in self.all_feature_names:
                if col not in df.columns:
                    df[col] = 0
            
            df = df[self.all_feature_names]
            
            # Apply the exact same preprocessing as training:
            # 1. Keep categorical columns as-is
            # 2. Standardize numerical columns using the fitted scaler
            
            X_processed = df.copy()
            
            # Apply standardization to numerical columns only (exactly like training)
            X_processed[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])
            
            # Make prediction
            risk_probability = 1- self.model.predict_proba(X_processed)[0][1]
            risk_class = self.model.predict(X_processed)[0]
            
            # Calculate traditional scores for interpretation
            traditional_scores = self._calculate_traditional_scores(features)
            
            # Determine risk level
            if risk_probability < 0.3:
                risk_level = "Low"
                risk_color = "success"
            elif risk_probability < 0.7:
                risk_level = "Moderate"
                risk_color = "warning"
            else:
                risk_level = "High"
                risk_color = "danger"
            
            # Generate interpretation
            interpretation = self._generate_interpretation(features, traditional_scores, risk_probability)
            
            return {
                'disease': 'HCC (Hepatocellular Carcinoma)',
                'risk_probability': risk_probability,
                'risk_percentage': risk_probability * 100,
                'risk_class': risk_class,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'traditional_scores': traditional_scores,
                'model_type': 'SVM Trained Model (Standardized)',
                'interpretation': interpretation,
                'has_afp': 'afp' in patient_data and patient_data['afp'] is not None and patient_data['afp'] > 0
            }
            
        except Exception as e:
            return {
                'disease': 'HCC (Hepatocellular Carcinoma)',
                'error': str(e),
                'risk_probability': 0.0,
                'risk_percentage': 0.0,
                'risk_class': 0,
                'risk_level': 'Error',
                'risk_color': 'secondary',
                'traditional_scores': {},
                'model_type': 'Error',
                'interpretation': f'Error in calculation: {str(e)}',
                'has_afp': False
            }
    
    def _calculate_traditional_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate traditional HCC-related scores"""
        scores = {}
        
        try:
            # FIB-4 Score
            age = features.get('Age', 0)
            ast = features.get('AST', 0)
            alt = features.get('ALT', 0)
            platelets = features.get('Trombosit', 0)  # Already scaled by 1000
            
            if platelets > 0 and alt > 0:
                scores['FIB-4'] = (age * ast) / (platelets * np.sqrt(alt))
            else:
                scores['FIB-4'] = 0
            
            # APRI Score
            if platelets > 0:
                scores['APRI'] = ((ast / 40) / platelets) * 100
            else:
                scores['APRI'] = 0
            
            # MELD Score
            inr = max(features.get('INR', 1), 1)
            total_bil = max(features.get('Total_Bil', 1), 1)
            creatinine = max(features.get('Creatinine', 1), 1)
            
            scores['MELD'] = (3.78 * np.log(total_bil) + 
                             11.2 * np.log(inr) + 
                             9.57 * np.log(creatinine) + 6.43)
            
            # AFP Risk Assessment
            afp = features.get('AFP', 0)
            if afp < 10:
                scores['AFP Risk'] = 'Low'
            elif afp < 200:
                scores['AFP Risk'] = 'Moderate'
            else:
                scores['AFP Risk'] = 'High'
            
        except Exception as e:
            print(f"Error calculating traditional scores: {e}")
        
        return scores
    
    def _generate_interpretation(self, features: Dict[str, Any], 
                                traditional_scores: Dict[str, float], 
                                risk_probability: float) -> str:
        """Generate clinical interpretation"""
        interpretation = []
        
        # Age factor
        age = features.get('Age', 0)
        if age > 60:
            interpretation.append("Advanced age increases HCC risk.")
        
        # Gender factor
        gender = features.get('Gender', 1)
        if gender == 2:  # Male
            interpretation.append("Male gender is associated with higher HCC risk.")
        
        # Liver enzyme assessment
        ast = features.get('AST', 0)
        alt = features.get('ALT', 0)
        if ast > 80 or alt > 80:
            interpretation.append("Significantly elevated liver enzymes suggest hepatocellular injury.")
        
        # Platelet assessment
        platelets = features.get('Trombosit', 0)
        if platelets < 150000:
            interpretation.append("Thrombocytopenia may indicate advanced liver disease.")
        
        # AFP assessment
        afp = features.get('AFP', 0)
        if afp > 400:
            interpretation.append("Very high AFP strongly suggests HCC.")
        elif afp > 200:
            interpretation.append("Elevated AFP is concerning for HCC.")
        elif afp > 20:
            interpretation.append("Mildly elevated AFP warrants monitoring.")
        
        # Traditional score interpretation
        if 'FIB-4' in traditional_scores:
            fib4 = traditional_scores['FIB-4']
            if fib4 > 3.25:
                interpretation.append("High FIB-4 suggests advanced fibrosis.")
        
        # Risk-based recommendations
        if risk_probability > 0.7:
            interpretation.append("High risk warrants immediate hepatology evaluation and imaging.")
        elif risk_probability > 0.3:
            interpretation.append("Moderate risk requires close monitoring and follow-up.")
        else:
            interpretation.append("Low risk but continue surveillance if risk factors present.")
        
        return " ".join(interpretation)


# Convenience function for easy import
def predict_hcc_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to predict HCC risk"""
    model = HCCRiskModelFinal()
    return model.predict_risk(patient_data)
