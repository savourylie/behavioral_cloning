import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras import initializations
from keras import backend as K

# def my_init(shape, name=None, dim_ordering='default'):
#     mu = 0
#     sigma = 0.15
#     value = np.random.normal(mu, sigma, shape)
#     return K.variable(value, name=name)

# initializations.my_init = my_init

# Fix error with Keras and TensorFlow

import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    from sklearn.decomposition import PCA
    import cv2

    def pca_gray_single_image(image):
        imshape = image.shape
        temp = image.reshape(imshape[0] * imshape[1], 3)

        pca = PCA(n_components=1, whiten=True)
        pca.fit(temp)
        
        temp2 = image.reshape(imshape[0] * imshape[1], 3)
        temp2 = pca.transform(temp2)
        temp2 = temp2.reshape(imshape[0], imshape[1])

        return temp2
        
    def crop_single_image(image):
        return image[70:][:][:-20][:]

    def crop_sky_and_front_cover(image_data):
        return np.apply_along_axis(crop_single_image, axis=1, arr=image_data)

    def normalize_single_image(image):
        return (image - np.mean(image)) / (np.max(image) - np.min(image))

    def rgb_2_hsv_single_image(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def preprocess(image_data):
        if len(image_data.shape) == 4:
            # Crop data
            cropped = crop_sky_and_front_cover(image_data)
            
            # Resize data
            image_data_small = []
            for image in cropped:
                image_data_small.append(cv2.resize(image, (0,0), fx=0.5, fy=0.5))
                
    #         image_data_pca = []
        
            # PCA Grayscale
    #         for image in image_data_small:
    #             image_data_pca.append(pca_gray_single_image(image))
            
    #         image_data_pca = np.asarray(image_data_pca)
            
            # HSV data
            image_data_hsv = []
            for image in image_data_small:
                image_data_hsv.append(rgb_2_hsv_single_image(image))
            
            image_data_normalized = []
        
            # Normalize data
            for image in image_data_hsv:
                image_data_normalized.append(normalize_single_image(image))
            
            image_data_normalized = np.asarray(image_data_normalized)
        
        elif len(image_data.shape) == 3:
            # Crop data
            cropped =  crop_single_image(image_data)
            
            # Resize data
            small = cv2.resize(cropped, (0,0), fx=0.5, fy=0.5)
            
            # PCA Grayscale
    #         image_data_pca = pca_gray_single_image(small)
            
            # HSV data
            image_data_hsv = rgb_2_hsv_single_image(small)
            
            # Normalize data
            image_data_normalized = normalize_single_image(image_data_hsv)
                                                     
        else:
            raise TypeError("Wrong image shape!")
        

        return image_data_normalized

    image_array = preprocess(image_array)


    # import matplotlib.pyplot as plt
    # plt.imshow(image_array, cmap='gray')
    # plt.show()

    # print(image_array)
    
    # import sys
    # sys.exit()

    # image_array = np.expand_dims(image_array, axis=2)

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.3
    # throttle = 1
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        # model = model_from_json(jfile.read())
        model = model_from_json(json.loads(jfile.read()))


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)