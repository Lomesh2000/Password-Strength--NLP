# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:05:05 2021

@author: lomes
"""
from flask import Flask,request,render_template,jsonify
#from model import word_to_char
import numpy as np
import pickle
import dill


app=Flask(__name__)
model=dill.load(open('model_dill','rb'))
vectoriser=dill.load(open('vectoriser_dill.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    password= request.form['password']
    password=[password]
    final_features = vectoriser.transform(password)
    prediction = model.predict(final_features)

    output = prediction[0]
    
    return render_template('result.html',prediction=output)

'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.predict(vectoriser.transform(data))

    output = prediction[0]
    
    if output==0:
        return render_template('home.html',prediction=output, prediction_text='Anyone can hack your password.Try a difficult one')
    elif output==1:
        return render_template('home.html',prediction=output,prediction_text='Not that safe.Try a more complex one')
    elif output==2:
        return render_template('home.html',prediction=output,prediction_text='very safe password')
'''
if __name__ == "__main__":
    app.run(debug=True)

