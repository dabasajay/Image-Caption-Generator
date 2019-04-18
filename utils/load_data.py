import numpy as np
from utils.preprocessing import *
from pickle import load, dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

'''
	*We have Flickr_8k.trainImages.txt and Flickr_8k.devImages.txt files which consist of unique identifiers(id) 
		which can be used to filter the images and their descriptions
	*Load a pre-defined list of photo identifiers(id)
'''
def load_set(filename):
	file = open(filename, 'r')
	doc = file.read()
	file.close()
	dataset = list()
	# Process line by line
	for line in doc.split('\n'):
		# Skip empty lines
		if len(line) < 1:
			continue
		# Get the image identifier(id)
		_id = line.split('.')[0]
		dataset.append(_id)
	return set(dataset)

'''
	*The model we'll develop will generate a caption for a given photo and the caption will be generated one word at a time. 
	*The sequence of previously generated words will be provided as input. Therefore, we will need a ‘first word’ to 
		kick-off the generation process and a ‘last word‘ to signal the end of the caption.
	*We'll use the strings ‘startseq‘ and ‘endseq‘ for this purpose. These tokens are added to the loaded descriptions
		as they are loaded. 
	*It is important to do this now before we encode the text so that the tokens are also encoded correctly.
	*Load clean descriptions into memory
'''
def load_clean_descriptions(filename, dataset):
	file = open(filename, 'r')
	doc = file.read()
	file.close()
	descriptions = dict()
	# Process line by line
	for line in doc.split('\n'):
		# Split line on white space
		tokens = line.split()
		# Split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# Skip images not in the set
		if image_id in dataset:
			# Create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# Wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions


'''
	*The description text will need to be encoded to numbers before it can be presented to the model.
	*The first step in encoding the data is to create a consistent mapping from words to unique integer values.
		Keras provides the Tokenizer class that can learn this mapping from the loaded description data.
'''
# Convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# Fit a tokenizer on given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

'''
	*Each description will be split into words. The model will be provided one word & the photo and it generates the next word. 
	*Then the first two words of the description will be provided to the model as input with the image to generate the next word. 
	*This is how the model will be trained.
	*For example, the input sequence “little girl running in field” would be 
		split into 6 input-output pairs to train the model:

		X1		X2(text sequence) 								y(word)
		-----------------------------------------------------------------
		photo	startseq, 									little
		photo	startseq, little,								girl
		photo	startseq, little, girl, 						running
		photo	startseq, little, girl, running, 				in
		photo	startseq, little, girl, running, in, 			field
		photo	startseq, little, girl, running, in, field, 	endseq
'''

# Create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
	# X1 : input for photo features
	# X2 : input for text features
	X1, X2, y = list(), list(), list()
	vocab_size = len(tokenizer.word_index) + 1
	# Walk through each description for the image
	for desc in desc_list:
		# Encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# Split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# Split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# Pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# Encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# Store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return np.array(X1), np.array(X2), np.array(y)

# Calculate the length of the description with the most words
def calc_max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# Load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# Data generator, intended to be used in a call to model.fit_generator()
def data_generator(photos, descriptions, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]

def loadTrainData(config,preprocessDataReady=True):
	train = load_set(config['train_data_path'])
	print('Training Dataset Length: %d' % len(train))
	# Check if we already have preprocessed data saved and if not, preprocess the data.
	if preprocessDataReady is False:
		preprocessData(config)
	# Descriptions
	train_descriptions = load_clean_descriptions(config['model_save_path']+'descriptions.txt', train)
	print('Descriptions for Training = %d' % len(train_descriptions))
	# Photo features
	train_features = load_photo_features(config['model_save_path']+'features.pkl', train)
	print('Photos for Training = %d' % len(train_features))
	# Prepare tokenizer
	tokenizer = create_tokenizer(train_descriptions)
	# Save the tokenizer
	dump(tokenizer, open(config['model_save_path']+'tokenizer.pkl', 'wb'))
	# Determine the maximum sequence length
	max_length = calc_max_length(train_descriptions)
	return train_features, train_descriptions, max_length

def loadValData(config):
	val = load_set(config['val_data_path'])
	print('Validation Dataset Length: %d' % len(val))
	# Descriptions
	val_descriptions = load_clean_descriptions(config['model_save_path']+'descriptions.txt', val)
	print('Descriptions for Validation = %d' % len(val_descriptions))
	# Photo features
	val_features = load_photo_features(config['model_save_path']+'features.pkl', val)
	print('Photos for Validation = %d' % len(val_features))
	return val_features, val_descriptions