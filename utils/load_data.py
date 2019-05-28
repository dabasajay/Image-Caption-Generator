import numpy as np
from utils.preprocessing import *
from pickle import load, dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import random
'''
	*We have Flickr_8k.trainImages.txt and Flickr_8k.devImages.txt files which consist of unique identifiers(id) 
		which can be used to filter the images and their descriptions
	*Load a pre-defined list of image identifiers(id)
	*Glimpse of file:
		2513260012_03d33305cf.jpg
		2903617548_d3e38d7f88.jpg
		3338291921_fe7ae0c8f8.jpg
		488416045_1c6d903fe0.jpg
		2644326817_8f45080b87.jpg
'''
def load_set(filename):
	file = open(filename, 'r')
	doc = file.read()
	file.close()
	ids = list()
	# Process line by line
	for line in doc.split('\n'):
		# Skip empty lines
		if len(line) < 1:
			continue
		# Get the image identifier(id)
		_id = line.split('.')[0]
		ids.append(_id)
	return set(ids)

'''
	*The model we'll develop will generate a caption for a given image and the caption will be generated one word at a time. 
	*The sequence of previously generated words will be provided as input. Therefore, we will need a ‘first word’ to 
		kick-off the generation process and a ‘last word‘ to signal the end of the caption.
	*We'll use the strings ‘startseq‘ and ‘endseq‘ for this purpose. These tokens are added to the captions
		as they are loaded. 
	*It is important to do this now before we encode the text so that the tokens are also encoded correctly.
	*Load captions into memory
	*Glimpse of file:
		1000268201_693b08cb0e child in pink dress is climbing up set of stairs in an entry way
		1000268201_693b08cb0e girl going into wooden building
		1000268201_693b08cb0e little girl climbing into wooden playhouse
		1000268201_693b08cb0e little girl climbing the stairs to her playhouse
		1000268201_693b08cb0e little girl in pink dress going into wooden cabin
'''
def load_cleaned_captions(filename, ids):
	file = open(filename, 'r')
	doc = file.read()
	file.close()
	captions = dict()
	_count = 0
	# Process line by line
	for line in doc.split('\n'):
		# Split line on white space
		tokens = line.split()
		# Split id from caption
		image_id, image_caption = tokens[0], tokens[1:]
		# Skip images not in the ids set
		if image_id in ids:
			# Create list
			if image_id not in captions:
				captions[image_id] = list()
			# Wrap caption in start & end tokens
			caption = 'startseq ' + ' '.join(image_caption) + ' endseq'
			# Store
			captions[image_id].append(caption)
			_count = _count+1
	return captions, _count

# Load image features
def load_image_features(filename, ids):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {_id: all_features[_id] for _id in ids}
	return features

# Convert a dictionary to a list
def to_lines(captions):
	all_captions = list()
	for image_id in captions.keys():
		[all_captions.append(caption) for caption in captions[image_id]]
	return all_captions

'''
	*The captions will need to be encoded to numbers before it can be presented to the model.
	*The first step in encoding the captions is to create a consistent mapping from words to unique integer values.
		Keras provides the Tokenizer class that can learn this mapping from the loaded captions.
	*Fit a tokenizer on given captions
'''
def create_tokenizer(captions):
	lines = to_lines(captions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# Calculate the length of the captions with the most words
def calc_max_length(captions):
	lines = to_lines(captions)
	return max(len(line.split()) for line in lines)

'''
	*Each caption will be split into words. The model will be provided one word & the image and it generates the next word. 
	*Then the first two words of the caption will be provided to the model as input with the image to generate the next word. 
	*This is how the model will be trained.
	*For example, the input sequence “little girl running in field” would be 
		split into 6 input-output pairs to train the model:

		X1		X2(text sequence) 								y(word)
		-----------------------------------------------------------------
		image	startseq,										little
		image	startseq, little,								girl
		image	startseq, little, girl,							running
		image	startseq, little, girl, running,				in
		image	startseq, little, girl, running, in,			field
		image	startseq, little, girl, running, in, field,		endseq
'''
# Create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, captions_list, image):
	# X1 : input for image features
	# X2 : input for text features
	# y  : output word
	X1, X2, y = list(), list(), list()
	vocab_size = len(tokenizer.word_index) + 1
	# Walk through each caption for the image
	for caption in captions_list:
		# Encode the sequence
		seq = tokenizer.texts_to_sequences([caption])[0]
		# Split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# Split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# Pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# Encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# Store
			X1.append(image)
			X2.append(in_seq)
			y.append(out_seq)
	return X1, X2, y

# Data generator, intended to be used in a call to model.fit_generator()
def data_generator(images, captions, tokenizer, max_length, batch_size, random_seed):
	# Setting random seed for reproducibility of results
	random.seed(random_seed)
	# Image ids
	image_ids = list(captions.keys())
	_count=0
	assert batch_size<= len(image_ids), 'Batch size must be less than or equal to {}'.format(len(image_ids))
	while True:
		if _count >= len(image_ids):
			# Generator exceeded or reached the end so restart it
			_count = 0
		# Batch list to store data
		input_img_batch, input_sequence_batch, output_word_batch = list(), list(), list()
		for i in range(_count, min(len(image_ids), _count+batch_size)):
			# Retrieve the image id
			image_id = image_ids[i]
			# Retrieve the image features
			image = images[image_id][0]
			# Retrieve the captions list
			captions_list = captions[image_id]
			# Shuffle captions list
			random.shuffle(captions_list)
			input_img, input_sequence, output_word = create_sequences(tokenizer, max_length, captions_list, image)
			# Add to batch
			for j in range(len(input_img)):
				input_img_batch.append(input_img[j])
				input_sequence_batch.append(input_sequence[j])
				output_word_batch.append(output_word[j])
		_count = _count + batch_size
		yield [[np.array(input_img_batch), np.array(input_sequence_batch)], np.array(output_word_batch)]

def loadTrainData(config):
	train_image_ids = load_set(config['train_data_path'])
	# Check if we already have preprocessed data saved and if not, preprocess the data.
	# Create and save 'captions.txt' & features.pkl
	preprocessData(config)
	# Load captions
	train_captions, _count = load_cleaned_captions(config['model_data_path']+'captions.txt', train_image_ids)
	# Load image features
	train_image_features = load_image_features(config['model_data_path']+'features_'+str(config['model_type'])+'.pkl', train_image_ids)
	print('{}: Available images for training: {}'.format(mytime(),len(train_image_features)))
	print('{}: Available captions for training: {}'.format(mytime(),_count))
	if not os.path.exists(config['model_data_path']+'tokenizer.pkl'):
		# Prepare tokenizer
		tokenizer = create_tokenizer(train_captions)
		# Save the tokenizer
		dump(tokenizer, open(config['model_data_path']+'tokenizer.pkl', 'wb'))
	# Determine the maximum sequence length
	max_length = calc_max_length(train_captions)
	return train_image_features, train_captions, max_length

def loadValData(config):
	val_image_ids = load_set(config['val_data_path'])
	# Load captions
	val_captions, _count = load_cleaned_captions(config['model_data_path']+'captions.txt', val_image_ids)
	# Load image features
	val_features = load_image_features(config['model_data_path']+'features_'+str(config['model_type'])+'.pkl', val_image_ids)
	print('{}: Available images for validation: {}'.format(mytime(),len(val_features)))
	print('{}: Available captions for validation: {}'.format(mytime(),_count))
	return val_features, val_captions