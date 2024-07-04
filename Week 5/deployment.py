# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'model.pkl')


from flask import Flask, request, jsonify
import numpy as np

# Create a Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Welcome to the Iris Classifier!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get data posted as a json
        prediction = model.predict(np.array(data['features']).reshape(1, -1))  # Predict using the model
        return jsonify({'prediction': int(prediction[0])})  # Return the prediction as JSON
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)