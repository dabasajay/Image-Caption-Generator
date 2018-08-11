from pickle import load
from utils.model import *
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# extract features from each photo in the directory
def extract_features(filename):
	model = defineCNNmodel()
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# load the tokenizer
tokenizer_path = 'model_data/tokenizer.pkl'
tokenizer = load(open(tokenizer_path, 'rb'))

# pre-define the max sequence length (from training)
max_length = 34

# load the model
model_path = 'model_data/model_19.h5'
model = load_model(model_path)

# load and prepare the photograph
test_path = 'test_data'
for image_file in os.listdir(test_path):
        try:
            image_type = imghdr.what(os.path.join(test_path, image_file))
            if not image_type:
                continue
        except IsADirectoryError:
            continue
image = extract_features(image_file)

# generate description
description = generate_desc(model, tokenizer, image, max_length)

# remove startseq and endseq
caption = 'Caption: ' + description.split()[1].capitalize()
for x in description.split()[2:len(description.split())-1]:
    caption = caption + ' ' + x
caption += '.'

# Show image and it's caption
pil_im = Image.open(image_file, 'r')
fig, ax = plt.subplots(figsize=(8, 8))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
_ = ax.imshow(np.asarray(pil_im), interpolation='nearest')
_ = ax.set_title(caption,fontdict={'fontsize': '20','fontweight' : '40'})