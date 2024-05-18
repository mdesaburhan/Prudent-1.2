import requests
import json

# Define sample data for prediction
sample_data = {
    "feature1": value1,
    "feature2": value2,
    # Add more features as needed
}

# Send POST request to the /predict endpoint
response = requests.post('http://127.0.0.1:5000/predict', json=sample_data)

# Check if request was successful (status code 200)
if response.status_code == 200:
    # Parse JSON response
    predictions = response.json()['predictions']
    print("Predictions:", predictions)
else:
    print("Error:", response.text)
