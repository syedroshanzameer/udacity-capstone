import os
import numpy as np
import json
import joblib
from azureml.core.model import Model

def init():
    global model
    try:
        model_path = Model.get_model_path('best_hd_run')
        model = joblib.load(model_path)
    except Exception as err:
        print("init method error: "+str(err))

def run(data):
    try:
        data = json.loads(data)
        data = np.array(data["data"])
        result = model.predict(data)
        return result.tolist()
    except Exception as err:
        return strn+"run method error: "+str(err)