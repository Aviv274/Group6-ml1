import pickle
import numpy as np

# Define file paths for the trained model and scaler
MODEL_PATH = "./logistic_regression_model.pkl"
SCALER_PATH = "./scaler.pkl"

# Load the trained logistic regression model
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    return model

# Load the scaler
def load_scaler():
    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

# Function to make a prediction
def predict_diabetes(patient_data):
    """
    patient_data: list of numerical values matching feature order [Pregnancies, Glucose, BloodPressure, SkinThickness,
                    Insulin, BMI, DiabetesPedigreeFunction, Age]
    Returns: Predicted class (0: No Diabetes, 1: Diabetes) and probability
    """
    model = load_model()
    scaler = load_scaler()
    
    # Convert input to numpy array and reshape for scaling
    patient_array = np.array(patient_data).reshape(1, -1)
    patient_scaled = scaler.transform(patient_array)
    
    # Make prediction
    prediction = model.predict(patient_scaled)
    prediction_proba = model.predict_proba(patient_scaled)
    
    return prediction[0], prediction_proba[0]

# Example usage
if __name__ == "__main__":
    sample_patient = [5, 120, 70, 25, 100, 30.0, 0.5, 40]  # Example input data
    result, probabilities = predict_diabetes(sample_patient)
    print(f"Predicted Outcome: {'Diabetic' if result == 1 else 'Not Diabetic'}")
    print(f"Probability: {probabilities}")