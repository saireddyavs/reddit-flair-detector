import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('page1.html')


@app.route('/automated_testing',method=['POST'])
def automate():
    

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)

    

    with open(filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    links = [x.strip() for x in content]

    answers=[get_data(x) for x in links]

    res = dict(zip(links, answers))



    if request.form['form_name'] === 'text_entry_form':
        # code to process textarea data
    else:
        # code to process file upload data

    



   
    

    return render_template('page2.html', prediction_text='Employee Salary should be $ {}'.format(int_features))


if __name__ == "__main__":
    app.run(debug=True)
