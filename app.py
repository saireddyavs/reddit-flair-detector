import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

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



@app.route('/automated_testing',methods=['POST'])
def automate():

    print("hi")
    
    content = request.files['file'].read().decode("utf-8")


    content=open(content,"rb").readlines()

    

    
    print(content)
    # you may also want to remove whitespace characters like `\n` at the end of each line
    links = [x.strip() for x in content]

       

    answers=[get_data(x.decode("utf-8"))[0] for x in links]

    res = dict(zip(links, answers))

    print(res)

    resp = jsonify(res)


    resp.status_code = 200
    print(resp)
    return resp

    
    

    
if __name__ == "__main__":
    app.run(debug=True)
