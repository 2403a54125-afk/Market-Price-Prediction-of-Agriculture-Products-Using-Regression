from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- MODEL LOADING ---
pipeline = None
try:
    # Load the trained model pipeline created by model.py
    pipeline = joblib.load('regression_model_pipeline.joblib')
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'regression_model_pipeline.joblib' not found. Please run model.py first.")
except Exception as e:
    print(f"ERROR: Failed to load model pipeline: {e}")

# --- WEB ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles both the GET request (displaying the form) and
    the POST request (processing the form and making a prediction).
    """
    prediction_result = None

    # Check if the model loaded and if the user submitted data
    if request.method == 'POST' and pipeline:
        try:
            # 1. Retrieve input data from the web form (POST request)
            product = request.form['product']
            month = request.form['month']
            year = int(request.form['year'])
            rainfall = float(request.form['rainfall'])
            wpi = float(request.form['wpi'])

            # 2. Structure the input data into a Pandas DataFrame
            # Column names MUST match the names used during model training in model.py
            input_data = pd.DataFrame({
                'Product': [product],
                'Month': [month],
                'Year': [year],
                'Rainfall': [rainfall],
                'WPI': [wpi]
            })

            # 3. Make the prediction using the loaded pipeline
            predicted_price = pipeline.predict(input_data)[0]

            # 4. Format the result
            prediction_result = f"Predicted Price: ₹{predicted_price:.2f}"

        except KeyError as e:
            # This catches errors if a required field is missing from the form
            prediction_result = f"Error: Missing form field - {e}. Ensure all fields are filled."
        except ValueError:
            # This catches errors if conversion to int/float fails
            prediction_result = "Error: Please enter valid numbers for Year, Rainfall, and WPI."
        except Exception as e:
            # General error handling
            prediction_result = f"An unexpected error occurred during prediction: {e}"

    # Render the index.html template, passing the prediction result (or None initially)
    return render_template('index.html', prediction=prediction_result)

# --- RUN THE APPLICATION ---
if __name__ == '__main__':
    # Setting debug=True restarts the server automatically on code changes
    # and provides helpful error messages.
    app.run(debug=True)