# MUHAMMAD USMAN NAVEED
# Frontend 
# Language used: Flask
# Description:
#   1. Loads trained models 
#   2. Gets input file from the user 
#   3. Gets the prediction from the model
#   4. Displays the result

from flask import Flask, render_template, request, redirect, url_for
import pickle
import librosa
import os
import numpy as np


app = Flask(__name__)


# PKL FILES
dt_pkl = "Decision Tree Model.pkl"
knn_pkl = "K-Nearest Neighbor.pkl"
rfc_pkl = "Random Forest Classifier.pkl"
svm_pkl = "Support Vector Machine.pkl"
gb_pkl = "Gradient Boosting.pkl"
lr_pkl = "Linear Regression.pkl"


def prediction_folder(p):
    if p == 0:
        print("A.C\n")
    if p == 1:
        print("Bulb-Light\n")
    if p == 2:
        print("Gaana-Song\n")
    if p == 3:
        print("T.V\n")


def prediction():

    dt_model = pickle.load(open(dt_pkl, 'rb'))
    knn_model = pickle.load(open(knn_pkl, 'rb'))
    rfc_model = pickle.load(open(rfc_pkl, 'rb'))
    svm_model = pickle.load(open(svm_pkl, 'rb'))
    gb_model = pickle.load(open(gb_pkl, 'rb'))
    lr_model = pickle.load(open(lr_pkl, 'rb'))

    downloadFolderPath = "C:\\Users\\Usman Naveed\\Downloads"
    os.chdir(downloadFolderPath)
    print("Path: ", os.getcwd())
    fileName = "recorded.wav"

    mfccList = []
    maxVal = 7600
    X , SR = librosa.load(fileName)
    mfcc1 = librosa.feature.mfcc(y=X, sr=SR)
    mfcc1 = np.mean(mfcc1.T, axis=0)
    mfcc1 = np.array(mfcc1.flatten())
    mfcc1 = np.pad(mfcc1, (0, maxVal - len(mfcc1)))
    mfccList.append(mfcc1)

    print(mfccList)
    print(len(mfccList))

    print("\nPREDICTIONS:\n")
    pred_dt = dt_model.predict(mfccList)
    print("Decision Tree: ", pred_dt)
    prediction_folder(pred_dt)

    pred_knn = knn_model.predict(mfccList)
    print("K-Nearest Neighbor: ", pred_knn)
    prediction_folder(pred_knn)

    pred_rfc = rfc_model.predict(mfccList)
    print("Random Forest Classfier: ", pred_rfc)
    prediction_folder(pred_knn)

    pred_svm = svm_model.predict(mfccList)
    print("Support Vector Machines: ", pred_svm)
    prediction_folder(pred_svm)

    pred_gb = gb_model.predict(mfccList)
    print("Gradient Boosting: ", pred_gb)
    prediction_folder(pred_gb)

    pred_lr = lr_model.predict(mfccList)
    print("Linear Regression: ", pred_lr)
    prediction_folder(pred_lr)

    # FOR DELETING THE DOWNLOADED FILE
    os.remove(fileName)

    return pred_dt[0], pred_knn[0], pred_rfc[0], pred_svm[0], pred_gb[0], pred_lr[0]

    # return '''<a href = "/NewPage"><h1><button onclick="getElementById('demo').innerHTML=Date()"></button></h1><a>'''


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/window')
def window():
    return render_template('ModelResults.html', pred=prediction())


if __name__ == '_app_':
    app.debug = True
    app.run(use_debug=True)

    