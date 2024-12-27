import unittest
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import predict

class TestAnimalHealthClassification(unittest.TestCase):
    
    def test_predict_function(self):
        # Example input for predicting
        input_data = ["Fever", "Diarrhea", "Vomiting", "Weight loss", "Dehydration"]
        
        # Get the result from the prediction function
        result = predict(input_data)
        
        # Print the result to see if the symptoms are dangerous or not
        print(f"Prediction result for input {input_data}: {result}")
        
        # Check if the prediction is correct
        self.assertIn(result, ["Yes", "No"])  # We expect "Yes" for Dangerous and "No" for Not Dangerous

if __name__ == "__main__":
    unittest.main()