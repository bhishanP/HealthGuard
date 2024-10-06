from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive JSON data
    data = request.get_json(force=True)
    
    # Ensure that the JSON data has the correct format for input features
    try:
        # Convert the JSON data into a DataFrame
        input_data = pd.DataFrame([data])
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Data formatting error'}), 400
    
    # Predict using the loaded model
    try:
        predictions = model.predict(input_data)
        # Convert prediction to list if it is in numpy array format
        predictions = predictions.tolist()
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Error during prediction'}), 500
    
    # Return the prediction in JSON format
    return jsonify({'prediction': predictions})

if __name__ == '__main__':
    app.run(debug=True)  
