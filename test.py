from flask import Flask , render_template, request, url_for
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

picfolder = os.path.join('static','images')
app.config['UPLOAD_FOLDER'] = picfolder

model = load_model("axie_model.h5")

x_vector = 120*120*3

image = load_img("./static/images/axie-full-transparent.png")
image = img_to_array(image)  
image = image.reshape(1, x_vector) 
image = image.astype('float32')
image /= 255
pre = model.predict(image)

class_name = ['aquatic', 'beast' , 'bird' , 'bug' , 'dawn' , 'dusk' , 'mech' , 'plant'  , 'reptile' ]
print(class_name[np.argmax(pre)])

imagefile = 'axie'

img_predict = os.path.join(app.config['UPLOAD_FOLDER'], imagefile)

print(img_predict)