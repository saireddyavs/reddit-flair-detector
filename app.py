import numpy as np
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify, render_template
import pickle

from preprocessor_and_predictor import get_data

from werkzeug import secure_filename



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


@app.route('/automated_testing',methods=['POST'])
def automate():


    #import urllib

    #url=request.files['file']

    #file=urllib.request.urlopen(url.read().decode("utf=8"))

    #for line in file:
    #        decoded_line=line
    #        print(decoded_line)
    
            

    
        

    print("hi")
    
    content = request.files['file']

    file=request.files['file']

    filename = secure_filename(content.filename) 

    import os


    file.save(os.path.join(os.getcwd(),filename))


    with open(os.path.join(os.getcwd(),filename)) as f:
            file_content = f.readlines()


    print(file_content)

    # print(request.get_json())

    # print(content)

    # print(content.filename)


    # content=content.read()

    

    
    print(type(file_content))
    # you may also want to remove whitespace characters like `\n` at the end of each line
    links = [x.strip() for x in file_content]


    print(links)
       

    answers=[get_data(x)[0] for x in links]

    res = dict(zip(links, answers))

    print(res)

    resp = jsonify(res)


    resp.status_code = 200
    print(resp)
    return resp

    
    

    
if __name__ == "__main__":
    app.run(debug=True)
