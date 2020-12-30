import sys
sys.path.append(r'/home/younesnd/Sentiment')
import pandas as pd 
import numpy as np 
import pickle
import flask
from tf_idf import *
from flask import Flask, request, render_template
import torch 
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")
@app.route('/predict' , methods = ['POST'])
def predict():
  text = request.form['content']
  text= [str(text)]
  pkl_filename = r'/home/younesnd/Downloads/pickle_model .pkl'
  with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
  damm= pd.DataFrame(['test'])
  damm["test"]= text
  exp = td_if_load.transform(damm["test"])
  predictions = pickle_model.predict(exp)
  result=()
  if predictions==2 :
    result= "Negative"
  elif predictions== 1 : 
    result = "Positive"
  else :
    result = "Neutre"
  return render_template('prediction.html' , preds = result)  
if __name__ == "__main__":
    app.run(host = 'localhost', port = 8000, debug=True)
