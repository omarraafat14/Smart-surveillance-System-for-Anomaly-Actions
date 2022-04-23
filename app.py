# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:27:46 2022

@author: omarr
"""
from flask import Flask ,request, render_template
from flask_restful import Resource, Api
import json
import numpy as np
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import prediction

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/result", methods=['GET','POST'])
def predict():
    if request.method == "GET" or  request.method == 'POST':
        url_link = request.form.get("vid_url", False)
        res = prediction.predict_single_action(url_link)
    return render_template("result.html" , vid_link = url_link, classes = list(res.keys()), props=list(res.values()))

if __name__ == "__main__":
    app.run(debug= True)

"""
#if some truble with tensorflow-gpu try that
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)
"""