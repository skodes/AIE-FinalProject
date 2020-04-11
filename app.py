import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil

from keras.applications.mobilenet_v2 import MobileNetV2

# Declare a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH_YOGA = 'models/Yoga_multiclass10epoch30.h5'

#lable names of all classes being used
label_names= ['backbend' ,'bigtoepose' ,'bridge' ,'childs' ,'cobrapose' ,'corpsepose' ,'cow_face_pose' ,'crocodilepose' ,'downwarddog' ,'durvasasana' ,'eaglepose' ,'easypose' ,'eight_angle_pose' ,'feathered_peacock' ,'fire_log_pose','fireflypose' ,'fishpose' ,'gatepose' ,'halfspinaltwist' ,'heropose' ,'lionpose' ,'lord_of_dance_pose' ,'lotus' ,'lunge' ,'monkeypose' ,'mountain' ,'noosepose' ,'peacockpose' ,'plank' ,'plowpose' ,'reclining_bound_angle_pose' ,'reclining_hand-to-big-toe _pose' ,'reclining_hero_pose' ,'scalepose' ,'seatedforwardbend' ,'side_reclining_leg_lift' ,'sideplank' ,'staffpose','standing_forward_bend' ,'standing_half_forward_bend' ,'supine_spinal_twist_pose' ,'thunderboltpose' ,'tree' ,'trianglepose' ,'turtlepose' ,'upward_plank' ,'warrior1' ,'warrior2' ,'wide-angle_seated_forward_bend' ,'yogic_sleep_pose']

# Load your own trained yoga model
model_yoga = load_model(MODEL_PATH_YOGA)
model_yoga._make_predict_function()

#load food prediction model
model_food= MobileNetV2(weights='imagenet')
model_food._make_predict_function()

print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((64, 64))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)

    #output = preds[0]

    return preds

def model_predict_food(img, model):
    img = img.resize((224, 224))
        # Preprocessing the image
    x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

        # Be careful how your trained model deals with the input
        # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/profile', methods=['GET'])
def profile():
    return render_template('profile.html')

@app.route('/journal', methods=['GET'])
def journal():
    return render_template('journal.html')

@app.route('/calorie', methods=['GET'])
def calorie():
    return render_template('calorie.html')

@app.route('/money', methods=['GET'])
def money():
    return render_template('money.html')

@app.route('/homepage', methods=['GET'])
def homepage():
    return render_template('homepage.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')





#yoga pose prediction
@app.route('/yoga', methods=['GET'])
def yoga():
    # Main page
    return render_template('index.html')

@app.route('/predictyoga', methods=['GET', 'POST'])
def predictyoga():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        preds = model_predict(img, model_yoga)

        result = (label_names[np.argmax(preds)])
        pred_proba = "{:.3f}".format(np.amax(preds))
        result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None

@app.route('/food', methods=['GET'])
def food():
    # Main page
    return render_template('food.html')

@app.route('/predictfood', methods=['GET', 'POST'])
def predictfood():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict_food(img, model_food)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None



if __name__ == '__main__':
    # app.run(port=5002, threaded=False)
    app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
