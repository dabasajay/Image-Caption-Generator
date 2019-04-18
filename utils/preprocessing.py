import numpy as np
import os
from pickle import dump
import string
from tqdm import tqdm
from utils.model import CNNModel
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

# The function returns a dictionary with key - image identifier(id) and value - image features.
def extract_features(path):
	# Get CNN Model from model.py
	model = CNNModel()
	features = dict()
	# Extract features from each photo
	for name in tqdm(os.listdir(path)):
		# load an image from file
		filename = path + name
		image = load_img(filename, target_size=(299, 299))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the Inceptionv3 model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
	return features

# Extract descriptions for images
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

# Save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def preprocessData(config):
	# Extract features from all images
	print('Generating image features...')
	features = extract_features(config['images_path'])
	print('Completed. Saving now...')
	# Save to file
	dump(features, open(config['model_save_path']+'features.pkl', 'wb'))
	print("Saved Successfully.")
	# Load descriptions containing file and parse descriptions
	descriptions = load_descriptions(config['descriptions_path'])
	print('Loaded Descriptions: %d ' % len(descriptions))
	# Clean descriptions
	clean_descriptions(descriptions)
	# Save descriptions
	save_descriptions(descriptions, config['model_save_path']+'descriptions.txt')
"""
	*Now descriptions.txt is of form :- id desc
		Example : 2252123185_487f21e336 stadium full of people watch game
"""