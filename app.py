from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the app
app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])  # 0 or 1
    age = int(request.form['age'])
    salary = float(request.form['salary'])

    input_data = np.array([[gender, age, salary]])  # ✅ 3 features
    scaled_data = scaler.transform(input_data)      # ✅ matches training

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        result = "User is likely to click the ad"
    else:
        result = "User is NOT likely to click the ad"

    return render_template("result.html", age=age, salary=salary, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
