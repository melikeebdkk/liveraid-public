"""
Cirrhosis Risk Assessment Model
Enhanced with real trained model and clinical scoring systems
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class CirrhosisRiskModel:
    def __init__(self):
        # Try to load the new XGBoost model first, fallback to old model
        self.model_path_xgb = os.path.join(os.path.dirname(__file__), 'cirrhosis_model_xgb.pkl')
        self.scaler_path_xgb = os.path.join(os.path.dirname(__file__), 'cirrhosis_scaler_xgb.pkl')
        self.imputer_path_xgb = os.path.join(os.path.dirname(__file__), 'cirrhosis_imputer_xgb.pkl')
        
        # Fallback to old model paths
        self.model_path = os.path.join(os.path.dirname(__file__), 'cirrhosis_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'cirrhosis_scaler.pkl')
        self.imputer_path = os.path.join(os.path.dirname(__file__), 'cirrhosis_imputer.pkl')
        
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
        
        self.model = None
        self.scaler = None
        self.imputer = None
        self.model_type = None
        
        # XGBoost model uses these feature names (from actual dataset)
        self.xgb_feature_names = [
            'age', 'gender', 'ast', 'alt', 'platelet', 'albumin', 'bmi', 
            'inr', 'total_bilirubin', 'creatin', 'direct_bilirubin', 'alp'
        ]
        
        # Legacy model feature names (for backward compatibility)
        self.legacy_feature_names = [
            'Age', 'Gender (Female=1, Male=2)', 'AST', 'ALT', 'Trombosit', 'Albumin', 
            'Body Mass Index', 'INR', 'Total Bilirubin', 'Creatinine', 'Direct Bilirubin', 'ALP'
        ]
        
        # Map form field names to model feature names
        self.field_mapping = {
            'age': 'age',
            'gender': 'gender',  # XGBoost model: 0=Female, 1=Male (matches notebook)
            'ast': 'ast',
            'alt': 'alt',
            'trombosit': 'platelet',  # Map trombosit -> platelet
            'albumin': 'albumin',
            'bmi': 'bmi',
            'inr': 'inr',
            'total_bilirubin': 'total_bilirubin',
            'creatinine': 'creatin',  # Map creatinine -> creatin
            'direct_bilirubin': 'direct_bilirubin',
            'alp': 'alp'
        }
        
        # Legacy field mapping for backward compatibility
        self.legacy_field_mapping = {
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
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing tools"""
        try:
            # Try to load XGBoost model first
            if (os.path.exists(self.model_path_xgb) and 
                os.path.exists(self.scaler_path_xgb) and 
                os.path.exists(self.imputer_path_xgb)):
                
                self.model = joblib.load(self.model_path_xgb)
                self.scaler = joblib.load(self.scaler_path_xgb)
                self.imputer = joblib.load(self.imputer_path_xgb)
                self.model_type = 'XGBoost'
                self.feature_names = self.xgb_feature_names
                print("✅ XGBoost cirrhosis model loaded successfully")
                
            # Fallback to legacy model
            elif (os.path.exists(self.model_path) and 
                  os.path.exists(self.scaler_path) and 
                  os.path.exists(self.imputer_path)):
                
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.imputer = joblib.load(self.imputer_path)
                self.model_type = 'Legacy'
                self.feature_names = self.legacy_feature_names
                print("✅ Legacy cirrhosis model loaded successfully")
                
            else:
                print("⚠️  Cirrhosis model files not found, using rule-based calculations")
                self.model = None
                self.model_type = 'Rule-based'
                
        except Exception as e:
            print(f"❌ Error loading cirrhosis model: {e}")
            self.model = None
            self.model_type = 'Rule-based'
    
    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict cirrhosis risk for a patient using the enhanced model
        
        Args:
            patient_data: Dictionary with patient parameters
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Map form fields to model features with proper conversion
            mapped_data = {}
            missing_fields = []
            
            # Use appropriate field mapping based on model type
            if self.model_type == 'XGBoost':
                field_mapping = self.field_mapping
            else:
                field_mapping = self.legacy_field_mapping
            
            for form_field, model_field in field_mapping.items():
                if form_field in patient_data:
                    value = patient_data[form_field]
                    
                    # NO gender conversion for XGBoost model - use raw values as-is
                    # The dataset has gender values 0, 1, 2 and we'll use them directly
                    
                    mapped_data[model_field] = value
                else:
                    missing_fields.append(form_field)
            
            print(f"📊 Cirrhosis Model ({self.model_type}) - Mapped feature data:")
            for field in self.feature_names:
                if field in mapped_data:
                    value = mapped_data[field]
                    if 'gender' in field.lower():
                        print(f"  {field}: {value} (raw dataset value - no conversion)")
                    else:
                        print(f"  {field}: {value}")
                else:
                    print(f"  {field}: MISSING")
            
            # Raise error if any required fields are missing
            if missing_fields:
                raise ValueError(f"Missing required fields for Cirrhosis prediction: {', '.join(set(missing_fields))}")
            
            # Prepare features in the correct order
            features = []
            for field in self.feature_names:
                value = float(mapped_data[field])
                features.append(value)
            
            print(f"🔢 Cirrhosis Model - Feature vector: {features}")
            
            # Convert to DataFrame for consistent processing
            X = pd.DataFrame([features], columns=self.feature_names)
            
            if self.model is not None:
                # Use trained model WITHOUT preprocessing for XGBoost
                if self.model_type == 'XGBoost':
                    # XGBoost model - use raw data without scaling/imputation
                    risk_probabilities = self.model.predict_proba(X)[0]
                    risk_class = self.model.predict(X)[0]
                    risk_probability = risk_probabilities[1]  # Probability of cirrhosis (class 1)
                    
                    model_type = f'Raw {self.model_type} Model (No Preprocessing)'
                    print(f"🎯 Cirrhosis Model - Raw XGBoost prediction: {risk_class}")
                    print(f"🎯 Cirrhosis Model - Raw XGBoost probabilities: {risk_probabilities}")
                    print(f"✅ Cirrhosis Model - Final risk probability: {risk_probability*100:.2f}%")
                    
                elif self.scaler is not None and self.imputer is not None:
                    # Legacy model - use preprocessing
                    X_imputed = self.imputer.transform(X)
                    X_scaled = self.scaler.transform(X_imputed)
                    
                    # Make prediction
                    risk_probabilities = self.model.predict_proba(X_scaled)[0]
                    risk_class = self.model.predict(X_scaled)[0]
                    risk_probability = risk_probabilities[1]  # Probability of cirrhosis (class 1)
                    
                    model_type = f'Enhanced {self.model_type} Trained Model'
                    print(f"🎯 Cirrhosis Model - Legacy prediction: {risk_class}")
                    print(f"🎯 Cirrhosis Model - Legacy probabilities: {risk_probabilities}")
                    print(f"✅ Cirrhosis Model - Final risk probability: {risk_probability*100:.2f}%")
                else:
                    # Fallback if preprocessing tools missing
                    risk_probabilities = self.model.predict_proba(X)[0]
                    risk_class = self.model.predict(X)[0]
                    risk_probability = risk_probabilities[1]
                    
                    model_type = f'Raw {self.model_type} Model'
                    print(f"🎯 Cirrhosis Model - Fallback prediction: {risk_class}")
                    print(f"🎯 Cirrhosis Model - Fallback probabilities: {risk_probabilities}")
                    print(f"✅ Cirrhosis Model - Final risk probability: {risk_probability*100:.2f}%")
                
            else:
                # Enhanced rule-based prediction with better scoring
                risk_probability, risk_class = self._enhanced_rule_based_prediction(mapped_data)
                model_type = 'Enhanced Rule-based Calculation'
            
            # Calculate traditional scores
            traditional_scores = self._calculate_traditional_scores(mapped_data)
            
            # Determine risk level and color
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
            interpretation = self._generate_interpretation(mapped_data, traditional_scores, risk_probability)
            
            return {
                'disease': 'Cirrhosis',
                'risk_probability': risk_probability,
                'risk_percentage': risk_probability * 100,
                'risk_class': risk_class,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'traditional_scores': traditional_scores,
                'model_type': model_type,
                'interpretation': interpretation,
                'confidence': abs(risk_probability - 0.5) * 2  # Confidence based on distance from 0.5
            }
            
        except Exception as e:
            print(f"❌ Error in cirrhosis prediction: {e}")
            return {
                'disease': 'Cirrhosis',
                'risk_probability': 0.5,
                'risk_percentage': 50.0,
                'risk_class': 0,
                'risk_level': "Unknown",
                'risk_color': "secondary",
                'traditional_scores': {},
                'model_type': 'Error Fallback',
                'interpretation': f"Error in prediction: {str(e)}",
                'confidence': 0.0
            }
    
    def _enhanced_rule_based_prediction(self, data: Dict[str, Any]) -> Tuple[float, int]:
        """
        Enhanced rule-based prediction when trained model is not available
        Based on clinical knowledge and the patterns from the training data
        """
        try:
            # Check for required fields - use field names based on model type
            if self.model_type == 'XGBoost':
                required_fields = ['age', 'ast', 'alt', 'platelet', 'albumin', 'inr', 'total_bilirubin']
                age_field = 'age'
                ast_field = 'ast'
                alt_field = 'alt'
                platelet_field = 'platelet'
                albumin_field = 'albumin'
                inr_field = 'inr'
                total_bil_field = 'total_bilirubin'
                bmi_field = 'bmi'
                alp_field = 'alp'
            else:
                # Legacy field names
                required_fields = ['Age', 'AST', 'ALT', 'Trombosit', 'Albumin', 'INR', 'Total Bilirubin']
                age_field = 'Age'
                ast_field = 'AST'
                alt_field = 'ALT'
                platelet_field = 'Trombosit'
                albumin_field = 'Albumin'
                inr_field = 'INR'
                total_bil_field = 'Total Bilirubin'
                bmi_field = 'Body Mass Index'
                alp_field = 'ALP'
            
            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)
            
            if missing_fields:
                raise ValueError(f"Missing required fields for rule-based prediction: {', '.join(missing_fields)}")
            
            # Key indicators for cirrhosis risk
            risk_score = 0.0
            
            # Age factor (higher risk with age, especially > 50)
            age = data[age_field]
            if age > 60:
                risk_score += 0.15
            elif age > 50:
                risk_score += 0.10
            elif age > 40:
                risk_score += 0.05
            
            # AST/ALT ratio (>1 suggests cirrhosis)
            ast = data[ast_field]
            alt = data[alt_field]
            if alt > 0:
                ast_alt_ratio = ast / alt
                if ast_alt_ratio > 2:
                    risk_score += 0.20
                elif ast_alt_ratio > 1.5:
                    risk_score += 0.15
                elif ast_alt_ratio > 1:
                    risk_score += 0.10
            
            # Platelet count (thrombocytopenia indicates portal hypertension)
            platelet = data[platelet_field]
            if platelet < 100:
                risk_score += 0.25
            elif platelet < 150:
                risk_score += 0.15
            elif platelet < 200:
                risk_score += 0.10
            
            # Albumin (hypoalbuminemia indicates liver dysfunction)
            albumin = data[albumin_field]
            if albumin < 3.0:
                risk_score += 0.20
            elif albumin < 3.5:
                risk_score += 0.15
            elif albumin < 4.0:
                risk_score += 0.10
            
            # INR (coagulopathy indicates liver dysfunction)
            inr = data[inr_field]
            if inr > 1.5:
                risk_score += 0.20
            elif inr > 1.3:
                risk_score += 0.15
            elif inr > 1.1:
                risk_score += 0.10
            
            # Total bilirubin (hyperbilirubinemia)
            total_bil = data[total_bil_field]
            if total_bil > 2.0:
                risk_score += 0.15
            elif total_bil > 1.5:
                risk_score += 0.10
            elif total_bil > 1.2:
                risk_score += 0.05
            
            # BMI (obesity can worsen liver disease) - optional field
            bmi = data.get(bmi_field, 25)  # BMI can have default as it's often calculated
            if bmi > 35:
                risk_score += 0.10
            elif bmi > 30:
                risk_score += 0.05
            
            # ALP elevation - optional field
            alp = data.get(alp_field, 100)  # ALP can have default as it's supplementary
            if alp > 200:
                risk_score += 0.10
            elif alp > 150:
                risk_score += 0.05
            
            # Normalize to probability (0-1)
            risk_probability = min(risk_score, 0.95)  # Cap at 95%
            risk_probability = max(risk_probability, 0.05)  # Floor at 5%
            
            # Determine class
            risk_class = 1 if risk_probability > 0.5 else 0
            
            print(f"🧮 Enhanced rule-based cirrhosis risk calculation: {risk_probability:.3f}")
            
            return risk_probability, risk_class
            
        except Exception as e:
            print(f"❌ Error in enhanced rule-based prediction: {e}")
            return 0.5, 0
    
    def _calculate_traditional_scores(self, mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate traditional clinical scores"""
        scores = {}
        
        try:
            # Check required fields for score calculations
            required_for_fib4 = ['Age', 'AST', 'ALT', 'Trombosit']
            required_for_apri = ['AST', 'Trombosit']
            required_for_meld = ['INR', 'Total Bilirubin', 'Creatinine']
            
            # FIB-4 Score
            if all(field in mapped_data for field in required_for_fib4):
                age = mapped_data['Age']
                ast = mapped_data['AST']
                alt = mapped_data['ALT']
                platelet = mapped_data['Trombosit']
                
                if platelet > 0 and alt > 0:
                    fib4 = (age * ast) / (platelet * (alt ** 0.5))
                    scores['FIB-4'] = round(fib4, 2)
                    
                    if fib4 < 1.45:
                        scores['FIB-4 Interpretation'] = "Low probability of advanced fibrosis"
                    elif fib4 < 3.25:
                        scores['FIB-4 Interpretation'] = "Intermediate probability - further evaluation needed"
                    else:
                        scores['FIB-4 Interpretation'] = "High probability of advanced fibrosis"
            else:
                missing_fib4 = [field for field in required_for_fib4 if field not in mapped_data]
                scores['FIB-4 Error'] = f"Missing fields: {', '.join(missing_fib4)}"
            
            # APRI Score
            if all(field in mapped_data for field in required_for_apri):
                ast = mapped_data['AST']
                platelet = mapped_data['Trombosit']
                
                if platelet > 0:
                    apri = (ast / 40) / (platelet / 1000) * 100
                    scores['APRI'] = round(apri, 2)
                    
                    if apri < 0.5:
                        scores['APRI Interpretation'] = "Low probability of significant fibrosis"
                    elif apri < 1.5:
                        scores['APRI Interpretation'] = "Intermediate probability"
                    else:
                        scores['APRI Interpretation'] = "High probability of significant fibrosis"
            else:
                missing_apri = [field for field in required_for_apri if field not in mapped_data]
                scores['APRI Error'] = f"Missing fields: {', '.join(missing_apri)}"
            
            # MELD Score (for liver disease severity)
            if all(field in mapped_data for field in required_for_meld):
                inr = mapped_data['INR']
                total_bil = mapped_data['Total Bilirubin']
                creatinine = mapped_data['Creatinine']
                
                meld = 3.78 * np.log(total_bil) + 11.2 * np.log(inr) + 9.57 * np.log(creatinine) + 6.43
                meld = max(6, min(40, meld))  # MELD score range 6-40
                scores['MELD'] = round(meld, 1)
                
                if meld < 10:
                    scores['MELD Interpretation'] = "Low mortality risk"
                elif meld < 15:
                    scores['MELD Interpretation'] = "Moderate mortality risk"
                elif meld < 20:
                    scores['MELD Interpretation'] = "High mortality risk"
                else:
                    scores['MELD Interpretation'] = "Very high mortality risk"
            else:
                missing_meld = [field for field in required_for_meld if field not in mapped_data]
                scores['MELD Error'] = f"Missing fields: {', '.join(missing_meld)}"
                
        except Exception as e:
            print(f"⚠️ Error calculating traditional scores: {e}")
            scores['Error'] = str(e)
        
        return scores
    
    def _generate_interpretation(self, mapped_data: Dict[str, Any], 
                                traditional_scores: Dict[str, Any], 
                                risk_probability: float) -> str:
        """Generate clinical interpretation"""
        try:
            interpretation = []
            
            # Risk level interpretation
            if risk_probability < 0.3:
                interpretation.append("Low cirrhosis risk based on current laboratory values.")
            elif risk_probability < 0.7:
                interpretation.append("Moderate cirrhosis risk detected. Enhanced monitoring recommended.")
            else:
                interpretation.append("High cirrhosis risk indicated. Immediate clinical evaluation advised.")
            
            # Key findings - only analyze if fields are present
            findings = []
            
            # Check AST/ALT ratio
            if 'AST' in mapped_data and 'ALT' in mapped_data:
                ast = mapped_data['AST']
                alt = mapped_data['ALT']
                if alt > 0:
                    ratio = ast / alt
                    if ratio > 1:
                        findings.append(f"AST/ALT ratio of {ratio:.2f} suggests possible liver damage")
            
            # Check platelet count
            if 'Trombosit' in mapped_data:
                platelet = mapped_data['Trombosit']
                if platelet < 150:
                    findings.append(f"Low platelet count ({platelet}) may indicate portal hypertension")
            
            # Check albumin
            if 'Albumin' in mapped_data:
                albumin = mapped_data['Albumin']
                if albumin < 3.5:
                    findings.append(f"Low albumin ({albumin}) suggests impaired liver synthesis")
            
            # Check INR
            if 'INR' in mapped_data:
                inr = mapped_data['INR']
                if inr > 1.3:
                    findings.append(f"Elevated INR ({inr}) indicates coagulopathy")
            
            if findings:
                interpretation.append("Key findings: " + "; ".join(findings))
            
            # Traditional scores summary
            if 'FIB-4' in traditional_scores:
                interpretation.append(f"FIB-4 score: {traditional_scores['FIB-4']} - {traditional_scores.get('FIB-4 Interpretation', '')}")
            
            return " ".join(interpretation)
            
        except Exception as e:
            return f"Error generating interpretation: {str(e)}"
        
        # Platelet assessment
        platelets = float(mapped_data['Trombosit'])
        if platelets < 100:
            interpretation.append("Thrombocytopenia may indicate portal hypertension.")
        elif platelets < 150:
            interpretation.append("Mild thrombocytopenia noted.")
        
        # Albumin assessment
        albumin = float(mapped_data['Albumin'])
        if albumin < 3.0:
            interpretation.append("Low albumin suggests impaired hepatic synthetic function.")
        elif albumin < 3.5:
            interpretation.append("Mild hypoalbuminemia noted.")
        
        # INR assessment
        inr = float(mapped_data['INR'])
        if inr > 1.5:
            interpretation.append("Elevated INR indicates coagulopathy.")
        elif inr > 1.2:
            interpretation.append("Mild coagulopathy noted.")
        
        # Traditional score interpretation
        if 'FIB-4' in traditional_scores:
            fib4 = traditional_scores['FIB-4']
            if fib4 < 1.45:
                interpretation.append("FIB-4 <1.45 suggests low probability of advanced fibrosis.")
            elif fib4 > 3.25:
                interpretation.append("FIB-4 >3.25 suggests high probability of advanced fibrosis.")
        
        if 'APRI' in traditional_scores:
            apri = traditional_scores['APRI']
            if apri < 0.5:
                interpretation.append("APRI <0.5 suggests low probability of significant fibrosis.")
            elif apri > 1.5:
                interpretation.append("APRI >1.5 suggests high probability of significant fibrosis.")
        
        if 'MELD' in traditional_scores:
            meld = traditional_scores['MELD']
            if meld < 10:
                interpretation.append("MELD <10 indicates low short-term mortality risk.")
            elif meld > 15:
                interpretation.append("MELD >15 indicates increased short-term mortality risk.")
        
        # Risk-based recommendations
        if risk_probability > 0.7:
            interpretation.append("High risk warrants urgent hepatology consultation.")
        elif risk_probability > 0.3:
            interpretation.append("Moderate risk - consider hepatology referral and regular monitoring.")
        else:
            interpretation.append("Low risk - routine monitoring and lifestyle modifications advised.")
        
        return " ".join(interpretation)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if model is available"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}


# Convenience function for easy import
def predict_cirrhosis_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to predict cirrhosis risk"""
    model = CirrhosisRiskModel()
    return model.predict_risk(patient_data)
