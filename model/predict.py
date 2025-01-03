import joblib
import pandas as pd

def predict(input_symptoms):
    # Use relative paths
    model = joblib.load("random_forest_model.pkl")
    encoder = joblib.load("symptoms_encoder.pkl")
    
    # Create a DataFrame with the same structure as training symptoms
    symptom_columns = ["symptoms1", "symptoms2", "symptoms3", "symptoms4", "symptoms5"]
    input_df = pd.DataFrame([input_symptoms], columns=symptom_columns)
    
    # Transform input symptoms using the encoder
    input_features = encoder.transform(input_df)
    
    # Predict
    predictions = model.predict(input_features)
    return "Yes" if predictions[0] == 1 else "No"

if __name__ == "__main__":
    # Example input
    example_symptoms = ["Fever", "Diarrhea", "Vomiting", "Weight loss", "Dehydration"]
    print("Prediction:", predict(example_symptoms))
