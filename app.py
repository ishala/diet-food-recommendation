from flask import Flask, render_template, request, jsonify
from recom import getRecommendations, defaultLevel
import pandas as pd

data = pd.read_csv('static/data/cleaned_diets.csv')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/get_items', methods=['GET'])
def get_items():
    diet_type = request.args.get('diet_type')
    items = data[data['tipe_diet'].str.contains(diet_type, case=False, na=False)]['resep_masakan'].unique().tolist()
    return jsonify(items)

@app.route('/submit', methods=['POST'])
def submit():
    formData = request.form.to_dict()

    # Mengatur nilai kosong menjadi None
    for key in ['protein', 'carbs', 'fat']:
        if not formData.get(key):
            formData[key] = None

    # Memeriksa apakah nilai protein, carbs, dan fat diisi
    if formData.get('protein') is None or formData.get('carbs') is None or formData.get('fat') is None:
        # Menggunakan fungsi defaultLevel untuk mengolah formData lebih lanjut
        formData = defaultLevel(formData)
    
    processedData = getRecommendations(formData)
    return render_template('result.html', data=processedData)

if __name__ == '__main__':
    app.run(debug=True)
