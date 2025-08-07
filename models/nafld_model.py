"""
NAFLD (Non-Alcoholic Fatty Liver Disease) Risk Assessment Model
Uses CatBoost model to classify between NAFL and NASH
Input fields: age, gender, AST, ALT, trombosit, albumin, bmi, inr, 
              total bilirubin, creatinine, direct bilirubin, ALP
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pickle
import os
import joblib


class NAFLDRiskModel:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'models-temp', 'nafld', 'catboost_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'nafld_scaler.pkl')
        self.imputer_path = os.path.join(os.path.dirname(__file__), 'nafld_imputer.pkl')
        self.model = None
        self.scaler = None
        self.imputer = None
        
        # Features expected by the CatBoost model (from training data)
        self.feature_names = [
            'Age', 'Gender (Female=1, Male=2)', 'AST', 'ALT', 'Trombosit', 'Albumin', 
            'Body Mass Index', 'INR', 'Total Bilirubin', 'Creatinine', 'Direct Bilirubin', 'ALP'
        ]
        
        # Initialize preprocessing tools
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
        self.load_model()
    
    def load_model(self):
        """Load the trained CatBoost model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("✅ NAFLD CatBoost model loaded successfully")
            else:
                print("⚠️  NAFLD CatBoost model file not found, using rule-based calculations")
                self.model = None
                
        except Exception as e:
            print(f"❌ Error loading NAFLD CatBoost model: {e}")
            self.model = None
    
    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict NAFLD classification (NAFL vs NASH) for a patient
        
        Args:
            patient_data: Dictionary with patient parameters
            
        Returns:
            Dictionary with prediction results
        """
        try:
            print("🔍 NAFLD Model - Raw patient data received:")
            for key, value in patient_data.items():
                print(f"  {key}: {value} (type: {type(value)})")
            
            # Map form field names to expected feature names
            field_mapping = {
                'age': 'Age',
                'gender': 'Gender (Female=1, Male=2)',
                'ast': 'AST',
                'alt': 'ALT',
                'trombosit': 'Trombosit',
                'albumin': 'Albumin',
                'bmi': 'Body Mass Index',
                'inr': 'INR',
                'total_bilirubin': 'Total Bilirubin',
                'creatinine': 'Creatinine',
                'direct_bilirubin': 'Direct Bilirubin',
                'alp': 'ALP'
            }
            
            # Map form fields to expected feature names
            mapped_data = {}
            missing_fields = []
            
            for form_field, feature_name in field_mapping.items():
                if form_field in patient_data:
                    mapped_data[feature_name] = patient_data[form_field]
            
            print("📊 NAFLD Model - Mapped feature data:")
            for feature_name in self.feature_names:
                if feature_name in mapped_data:
                    value = mapped_data[feature_name]
                    print(f"  {feature_name}: {value}")
                else:
                    print(f"  {feature_name}: MISSING")
                    missing_fields.append(feature_name)
            
            # Raise error if any required fields are missing
            if missing_fields:
                raise ValueError(f"Missing required fields for NAFLD prediction: {', '.join(missing_fields)}")
            
            # Convert to numeric values
            features = []
            for field in self.feature_names:
                value = float(mapped_data[field])
                features.append(value)
            
            print(f"🔢 NAFLD Model - Final feature vector: {features}")
            
            # Convert to DataFrame for consistent processing
            X = pd.DataFrame([features], columns=self.feature_names)
            
            if self.model is not None:
                print("🤖 NAFLD Model - Using trained CatBoost model")
                # Use trained CatBoost model WITHOUT preprocessing
                # CatBoost can handle raw features directly
                
                print(f"📊 NAFLD Model - Raw input data shape: {X.shape}")
                print(f"📊 NAFLD Model - Raw input data: {X.iloc[0].tolist()}")
                
                # Make prediction directly on raw data
                prediction = self.model.predict(X)[0]
                probabilities = self.model.predict_proba(X)[0]
                
                print(f"🎯 NAFLD Model - Raw prediction: {prediction}")
                print(f"🎯 NAFLD Model - Probabilities: {probabilities}")
                print(f"🎯 NAFLD Model - Probabilities shape: {probabilities.shape}")
                
                # Determine classification
                if prediction == 1:
                    classification = "NAFL"
                    classification_description = "Non-Alcoholic Steatohepatitis (Inflammatory)"
                    
                    risk_color = "warning"
                    confidence = probabilities[0] * 100  # Probability for class 1 (NAFL)
                else:  # prediction == 2
                    classification = "NASH"
                    classification_description = "Non-Alcoholic Fatty Liver (Simple Steatosis)"
                    risk_color = "danger"
                    confidence = probabilities[1] * 100  # Probability for class 2 (NASH)
                
                print(f"✅ NAFLD Model - Final classification: {classification}")
                print(f"✅ NAFLD Model - Final confidence: {confidence}%")
                
                model_type = 'CatBoost Trained Model'
            else:
                print("🔄 NAFLD Model - Using rule-based fallback")
                # Fallback to rule-based prediction
                classification, classification_description, risk_color, confidence = self._mock_classification(patient_data)
                model_type = 'Rule-based Classification'
                print(f"✅ NAFLD Model - Rule-based result: {classification} ({confidence}%)")
            
            # Calculate traditional scores
            traditional_scores = self._calculate_traditional_scores(patient_data)
            
            # Generate interpretation
            interpretation = self._generate_interpretation(patient_data, traditional_scores, classification)
            
            return {
                'disease': 'MAFLD Classification',
                'classification': classification,
                'classification_description': classification_description,
                'confidence': confidence,
                'prediction_class': prediction if self.model else 1,
                'risk_color': risk_color,
                'traditional_scores': traditional_scores,
                'model_type': model_type,
                'interpretation': interpretation
            }
            
        except Exception as e:
            return {
                'disease': 'MAFLD Classification',
                'error': str(e),
                'classification': 'Error',
                'classification_description': 'Error in classification',
                'confidence': 0.0,
                'prediction_class': 0,
                'risk_color': 'secondary',
                'traditional_scores': {},
                'model_type': 'Error',
                'interpretation': f'Error in calculation: {str(e)}'
            }
    
    def _mock_classification(self, patient_data: Dict[str, Any]) -> tuple:
        """
        Mock classification for NAFL vs NASH when model is not available
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            Tuple with (classification, description, color, confidence)
        """
        # Rule-based classification using clinical indicators
        score = 0.0
        
        # Age factor (higher age increases NASH risk)
        age = float(patient_data.get('Age', patient_data.get('age', 0)))
        if age > 50:
            score += 0.3
        elif age > 40:
            score += 0.2
        
        # BMI factor (higher BMI increases NASH risk)
        bmi = float(patient_data.get('BMI', patient_data.get('bmi', 0)))
        if bmi > 35:
            score += 0.4
        elif bmi > 30:
            score += 0.3
        elif bmi > 25:
            score += 0.2
        
        # Liver enzyme levels (higher levels suggest NASH)
        ast = float(patient_data.get('AST', patient_data.get('ast', 0)))
        alt = float(patient_data.get('ALT', patient_data.get('alt', 0)))
        if ast > 80 or alt > 80:
            score += 0.4
        elif ast > 40 or alt > 40:
            score += 0.3
        elif ast > 30 or alt > 30:
            score += 0.2
        
        # AST/ALT ratio (higher ratio may indicate NASH)
        if ast > 0 and alt > 0:
            ast_alt_ratio = ast / alt
            if ast_alt_ratio > 1.5:
                score += 0.3
            elif ast_alt_ratio > 1.0:
                score += 0.2
        
        # Platelet count (lower platelets may indicate NASH)
        platelets = float(patient_data.get('Trombosit', patient_data.get('trombosit', 0)))
        if platelets < 150:
            score += 0.2
        elif platelets < 100:
            score += 0.3
        
        # Albumin (low albumin may indicate NASH)
        albumin = float(patient_data.get('Albumin', patient_data.get('albumin', 0)))
        if albumin < 3.5:
            score += 0.2
        elif albumin < 4.0:
            score += 0.1
        
        # INR (elevated in advanced disease)
        inr = float(patient_data.get('INR', patient_data.get('inr', 0)))
        if inr > 1.3:
            score += 0.2
        elif inr > 1.1:
            score += 0.1
        
        # Classify based on score
        if score > 0.6:
            return "NASH", "Non-Alcoholic Steatohepatitis (Inflammatory)", "danger", min(score * 100, 95.0)
        else:
            return "NAFL", "Non-Alcoholic Fatty Liver (Simple Steatosis)", "warning", min((1-score) * 100, 95.0)
    
    def _calculate_traditional_scores(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate traditional NAFLD scores"""
        scores = {}
        
        try:
            # NAFLD Fibrosis Score (NFS)
            age = float(patient_data.get('Age', patient_data.get('age', 0)))
            bmi = float(patient_data.get('BMI', patient_data.get('bmi', 0)))
            diabetes = 0  # Assuming no diabetes data, could be enhanced
            ast = float(patient_data.get('AST', patient_data.get('ast', 0)))
            alt = float(patient_data.get('ALT', patient_data.get('alt', 0)))
            platelets = max(float(patient_data.get('Trombosit', patient_data.get('trombosit', 0))), 1)
            albumin = float(patient_data.get('Albumin', patient_data.get('albumin', 0)))
            
            # AST/ALT ratio
            ast_alt_ratio = ast / max(alt, 1)
            
            # NFS calculation
            nfs = (-1.675 + 0.037 * age + 0.094 * bmi + 
                   1.13 * diabetes + 0.99 * ast_alt_ratio - 
                   0.013 * platelets - 0.66 * albumin)
            scores['NFS'] = nfs
            
            # FIB-4 Score (also applicable to NAFLD)
            fib4 = (age * ast) / (platelets * np.sqrt(alt))
            scores['FIB-4'] = fib4
            
            # APRI Score 
            apri = ((ast / 40) / platelets) * 100
            scores['APRI'] = apri
            
            # BMI Score (important for NAFLD)
            if bmi < 18.5:
                bmi_category = "Underweight"
            elif bmi < 25:
                bmi_category = "Normal"
            elif bmi < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"
            
            scores['BMI_Category'] = bmi_category
            
        except Exception as e:
            print(f"Error calculating traditional scores: {e}")
        
        return scores
    
    def _generate_interpretation(self, patient_data: Dict[str, Any], 
                                traditional_scores: Dict[str, float], 
                                classification: str) -> str:
        """Generate clinical interpretation for NAFL vs NASH classification"""
        interpretation = []
        
        # Classification-based interpretation
        if classification == "NASH":
            interpretation.append("NASH (Non-Alcoholic Steatohepatitis) is characterized by liver inflammation and may progress to fibrosis.")
            interpretation.append("This condition requires active monitoring and intervention.")
        else:
            interpretation.append("NAFL (Non-Alcoholic Fatty Liver) is simple steatosis without significant inflammation.")
            interpretation.append("This is a milder form but still requires lifestyle modifications.")
        
        # BMI assessment
        bmi = float(patient_data.get('BMI', patient_data.get('bmi', 0)))
        if bmi >= 30:
            interpretation.append("Obesity (BMI ≥30) significantly increases progression risk.")
        elif bmi >= 25:
            interpretation.append("Overweight status (BMI 25-29.9) is a moderate risk factor.")
        
        # Liver enzyme assessment
        ast = float(patient_data.get('AST', patient_data.get('ast', 0)))
        alt = float(patient_data.get('ALT', patient_data.get('alt', 0)))
        if ast > 40 or alt > 40:
            interpretation.append("Elevated liver enzymes suggest hepatic inflammation.")
        
        # AST/ALT ratio interpretation
        if ast > 0 and alt > 0:
            ast_alt_ratio = ast / alt
            if ast_alt_ratio > 1.5:
                interpretation.append("AST/ALT ratio >1.5 may indicate more advanced disease.")
        
        # Traditional score interpretation
        if 'NFS' in traditional_scores:
            nfs = traditional_scores['NFS']
            if nfs < -1.455:
                interpretation.append("NFS <-1.455 suggests low probability of advanced fibrosis.")
            elif nfs > 0.676:
                interpretation.append("NFS >0.676 suggests high probability of advanced fibrosis.")
            else:
                interpretation.append("NFS in intermediate range - further evaluation may be needed.")
        
        if 'FIB-4' in traditional_scores:
            fib4 = traditional_scores['FIB-4']
            if fib4 < 1.30:
                interpretation.append("FIB-4 <1.30 suggests low risk of advanced fibrosis.")
            elif fib4 > 2.67:
                interpretation.append("FIB-4 >2.67 suggests high risk of advanced fibrosis.")
        
        # Classification-based recommendations
        if classification == "NASH":
            interpretation.append("NASH requires lifestyle intervention, regular monitoring, and possible medical treatment.")
        else:
            interpretation.append("NAFL management focuses on lifestyle modifications and monitoring for progression.")
        
        return " ".join(interpretation)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if model is available"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}


# Convenience function for easy import
def predict_nafld_classification(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to predict NAFLD classification (NAFL vs NASH)"""
    model = NAFLDRiskModel()
    return model.predict_risk(patient_data)
