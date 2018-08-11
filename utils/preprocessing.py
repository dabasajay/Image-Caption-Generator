import numpy as np
import string
from os import listdir
from pickle import dump
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

#The function returns a dictionary of image identifier to image features.
def extract_features(path):
 	model = defineCNNmodel()
 	# extract features from each photo
 	features = dict()
 	for name in listdir(path):
 		# load an image from file
 		filename = path + '/' + name
 		image = load_img(filename, target_size=(299, 299))
 		# convert the image pixels to a numpy array
 		image = img_to_array(image)
 		# reshape data for the model
 		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
 		# prepare the image for the VGG model
 		image = preprocess_input(image)
 		# get features
 		feature = model.predict(image, verbose=0)
 		# get image id
 		image_id = name.split('.')[0]
 		# store feature
 		features[image_id] = feature
	return features

# extract descriptions for images
def load_descriptions(filename):
	file = open(filename, 'r')
	doc = file.read()
	file.close()
	mapping = dict()
	# process lines by line
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def preprocessData():
	# extract features from all images
	path = 'Flicker8k_Dataset'
	print('Generating image features...')
	features = extract_features(path)
	print('Completed. Saving now...')
	# save to file
	dump(features, open('model_data/features.pkl', 'wb'))
	print("Save Complete.")

	# load descriptions containing file and parse descriptions
	descriptions_path = 'train_val_data/Flickr8k.token.txt'

	descriptions = load_descriptions(descriptions_path)
	print('Loaded Descriptions: %d ' % len(descriptions))

	# clean descriptions
	clean_descriptions(descriptions)

	# save descriptions
	save_descriptions(descriptions, 'model_data/descriptions.txt')


# Now descriptions.txt is of form :
# Example : 2252123185_487f21e336 stadium full of people watch game