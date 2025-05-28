from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = pickle.load(open('milk_quality_model.pkl', 'rb'))
scaler = pickle.load(open('milk_quality_scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data input
    features = [float(request.form.get(feat)) for feat in ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']]
    
    # Ubah ke numpy array dan skalakan
    final_input = scaler.transform([features])
    
    # Prediksi
    prediction = model.predict(final_input)[0]

    # Map ke label
    label_map = {1: 'Low', 2: 'Medium', 3: 'High'}
    result = label_map.get(prediction, "Unknown")

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
