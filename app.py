from flask import Flask , render_template, request, url_for
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from werkzeug.utils import redirect
import os

app = Flask(__name__)

picfolder = os.path.join('static','images')
app.config['UPLOAD_FOLDER'] = picfolder

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    print(imagefile)
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)

    model = load_model("axie_model.h5")

    x_vector = 120*120*3

    image = load_img(image_path)
    image = img_to_array(image)  
    image = image.reshape(1, x_vector) 
    image = image.astype('float32')
    image /= 255
    pre = model.predict(image)

    class_name = ['aquatic', 'beast' , 'bird' , 'bug' , 'dawn' , 'dusk' , 'mech' , 'plant'  , 'reptile' ]
    result = class_name[np.argmax(pre)]


    img_predict = os.path.join(app.config['UPLOAD_FOLDER'], str(imagefile.filename))

    return render_template('index.html', prediction=result , img_predict = img_predict)

@app.route('/display/<filename>')
def display(filename):
    return redirect( url_for ('static', filename = 'images/' + imagefile.filename ), code = 301)

if __name__ == '__main__':
    app.run(port=3000, debug=True)