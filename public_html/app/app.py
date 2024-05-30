from flask import Flask, render_template, request
from recom import getRecommendations

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/submit', methods=['POST'])
def submit():
    formData = request.form.to_dict()    
    
    processedData = getRecommendations(formData)
    return render_template('result.html', data=processedData)


if __name__ == '__main__':
    app.run(debug=True)