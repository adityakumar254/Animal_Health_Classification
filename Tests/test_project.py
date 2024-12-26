import os
import sys
import pytest
# Add project directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.train_model import train_model
from model.predict import predict
from model.evaluate import evaluate_model

@pytest.fixture(scope="module")
def train_model_fixture():
    print("\n[INFO] Running train_model()...")
    train_model()
    assert os.path.exists("random_forest_model.pkl"), "Model file not found after training."
    assert os.path.exists("symptoms_encoder.pkl"), "Encoder file not found after training."
    print("[INFO] Model and encoder files generated successfully.")
    return True

def test_train_model(train_model_fixture):
    assert train_model_fixture, "Training setup failed."
    print("[SUCCESS] Training test passed.")

def test_predict(train_model_fixture):
    # Use full path to model files if necessary
    example_symptoms = ["Fever", "Diarrhea", "Vomiting", "Weight loss", "Dehydration"]
    prediction = predict(example_symptoms)
    assert prediction in ["Yes", "No"], "Prediction output is invalid."
    print(f"[SUCCESS] Prediction test passed. Output: {prediction}")

def test_evaluate_model(train_model_fixture):
    try:
        evaluate_model()
        print("[SUCCESS] Evaluation test passed.")
    except Exception as e:
        assert False, f"[ERROR] Evaluation failed with error: {e}"
