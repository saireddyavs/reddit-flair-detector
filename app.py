
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

    answer=get_data([link])[link]
    return render_template('page2.html', prediction_text=answer)


@app.route('/automated_testing',methods=['POST'])
def automate():


    #import urllib

    #url=request.files['file']

    #file=urllib.request.urlopen(url.read().decode("utf=8"))

    #for line in file:
    #        decoded_line=line
    #        print(decoded_line)
    
            

    
        

 
    
    content = request.files['upload_file']

    file=request.files['upload_file']

    filename = secure_filename(content.filename) 

    import os


    file.save(os.path.join(os.getcwd(),filename))


    with open(os.path.join(os.getcwd(),filename)) as f:
            file_content = f.readlines()


   

    # print(request.get_json())

    # print(content)

    # print(content.filename)


    # content=content.read()

    

  
    # you may also want to remove whitespace characters like `\n` at the end of each line
    links = [x.strip() for x in file_content]



    answers=get_data(links)

    

  
    resp = jsonify(answers)


    resp.status_code = 200
    
    return resp

    
    

    
if __name__ == "__main__":
    app.run(debug=True)
