from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained decision tree model
model = joblib.load('decision_tree_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        SeniorCitizen = int(request.form['SeniorCitizen'])
        PhoneService = int(request.form['PhoneService'])
        PaperlessBilling = int(request.form['PaperlessBilling'])
        MonthlyCharges = float(request.form['MonthlyCharges'])

        # Create DataFrame with user input
        input_data = pd.DataFrame({
            'SeniorCitizen': [SeniorCitizen],
            'PhoneService': [PhoneService],
            'PaperlessBilling': [PaperlessBilling],
            'MonthlyCharges': [MonthlyCharges]
        })

        # Make probability predictions
        probabilities = model.predict_proba(input_data)

        # Return probabilities as HTML response
        churn_probability = probabilities[0][1] * 100  # Probability of churn
        return render_template('result.html', churn_probability=churn_probability)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
