import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("models/pm_classificacao.h5")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(data: dict):

    new_sample = np.array(list(data.values()))
    new_sample_scaled = scaler.transform(new_sample)

    predictions = model.predict(new_sample_scaled)[0]

    probabilities = {'Heat Dissipation'     : predictions[0],
                     'No Failure'           : predictions[1],
                     'Overstrain Failure'   : predictions[2],
                     'Power Failure'        : predictions[3],
                     'Random Failures'      : predictions[4],
                     'Tool Wear Failure'    : predictions[5]
                     }   

    return probabilities