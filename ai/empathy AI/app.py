from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__, static_url_path='/static')
# model = load_model('E:\Assets of Youth\empathy ai\depression_model.h5')
model = keras.models.load_model('E:\Assets of Youth\empathy AI\depression_model.h5')


def preprocess_text(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    max_sequence_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    text = request.form['text']
    padded_text = preprocess_text(text)
    prediction = model.predict(padded_text)[0][0]
    pred = f"{prediction:.4f}"
    result = f"The depression score for the given text is: {prediction:.4f}"
    return render_template('index.html', prediction=pred, prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
