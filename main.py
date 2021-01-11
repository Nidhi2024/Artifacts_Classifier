import flask
from flask import Flask, redirect, url_for, request, render_template
import werkzeug
#keras
import keras
import keras.models
from keras.layers import LeakyReLU
from keras.applications.vgg19 import VGG19
import numpy as np
import cv2

app = flask.Flask(__name__)
def model_predict(img_path, model):
    IMG_SIZE = 224
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    

    # Preprocessing the image
    # resizing the image for preprocessing
    x1 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # x = np.true_divide(x, 255)
    ## Scaling

   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = x1.reshape(-1, IMG_SIZE, IMG_SIZE,3)

    preds = model.predict(x)
    preds= np.argmax(preds,axis=-1)
    if preds[0]==0:
        preds="Basket"
    elif preds[0]==1:
        preds="Coin"
    else:
        preds="Figure"
    
    
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    

    loaded_model = keras.models.load_model('models/vgg19.h5', custom_objects={'LeakyReLU': LeakyReLU})
    predicted_label = model_predict(filename, loaded_model)
    #preds = model_predict(file_path, model)
    result=predicted_label
    return result
    return None

app.run(host="127.0.0.1", port=5000, debug=True)