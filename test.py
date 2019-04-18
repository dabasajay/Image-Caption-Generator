import numpy as np
from PIL import Image
from pickle import load
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input

from config import config

# extract features from each photo in the directory
def extract_features(filename,model):
	# load the photo
	image = load_img(filename, target_size=(299, 299))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the Inceptionv3 model
	image = preprocess_input(image)
	# get features
	features = model.predict(image, verbose=0)
	return features

# load the tokenizer
tokenizer_path = config['tokenizer_path']
tokenizer = load(open(tokenizer_path, 'rb'))

# max sequence length (from training)
max_length = config['max_length']

# load the model from last epoch
model_path = config['model_save_path']+'model_epoch_'+str(config['num_of_epochs']-1)+'.h5'
caption_model = load_model(model_path)

image_model = CNNModel()

# load and prepare the photograph
for image_file in os.listdir(config['test_data_path']):
	if(image_file.split('.')[1]=='jpg' or image_file.split('.')[1]=='jpeg'):
		image = extract_features(config['test_data_path']+image_file,image_model)
		# generate description
		description = generate_desc(caption_model, tokenizer, image, max_length)
		# remove startseq and endseq
		caption = 'Caption: ' + description.split()[1].capitalize()
		for x in description.split()[2:len(description.split())-1]:
		    caption = caption + ' ' + x
		caption += '.'
		# Show image and it's caption
		pil_im = Image.open(config['test_data_path']+image_file, 'r')
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		_ = ax.imshow(np.asarray(pil_im), interpolation='nearest')
		_ = ax.set_title(caption,fontdict={'fontsize': '20','fontweight' : '40'})
		plt.savefig(config['test_data_path']+'output_'+image_file)