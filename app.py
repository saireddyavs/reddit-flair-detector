import numpy as np
from flask import Flask, request, jsonify, render_template


from preprocessor_and_predictor import get_data

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('page1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(request.form['search'])
    link=request.form['search']

    answer=get_data(link)[0]

    
    

    return render_template('page2.html', prediction_text=answer)


if __name__ == "__main__":
    app.run(debug=True)
