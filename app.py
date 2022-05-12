from flask import Flask, render_template, request
import tensorflow
import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index(): 
    return render_template("index.html")

# intializing label dictonary
label_dict = {0 : 'HEALTHY', 1 : 'MULTIPLE DISEASES', 2 : 'RUST', 3 : 'SCAB'}

# loading model weights
model = keras.models.load_model("best_model_efficientnetb4.hdf5")

def predict_class(img_path):
	img = load_img(img_path, target_size=(256, 256))
	img = img_to_array(img)/255
	img = img.reshape(1, 256, 256, 3)
	predict = model.predict(img)
	label = label_dict[int(predict.argmax(axis = 1))]
	return label

@app.route("/predict", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['image']
		img_path = "static/" + img.filename	
		img.save(img_path)
		predict = predict_class(img_path)
	return render_template("index.html", prediction = predict , img_path = img_path)

if __name__ == '__main__':
    app.run(debug=True)