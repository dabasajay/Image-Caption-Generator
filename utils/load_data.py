from utils import preprocessing

# Load image data and text data that we can use to fit the model

# We have Flickr_8k.trainImages.txt and Flickr_8k.devImages.txt files which consist
# of unique identifiers which can be used to filter the images and their descriptions

# load a pre-defined list of photo identifiers
def load_set(filename):
	file = open(filename, 'r')
	doc = file.read()
	file.close()
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

'''
The model we will develop will generate a caption given a photo, 
and the caption will be generated one word at a time. 
The sequence of previously generated words will be provided as input. 
Therefore, we will need a ‘first word’ to kick-off the generation process 
and a ‘last word‘ to signal the end of the caption.

We will use the strings ‘startseq‘ and ‘endseq‘ for this purpose. 
These tokens are added to the loaded descriptions as they are loaded. 
It is important to do this now before we encode the text so that 
the tokens are also encoded correctly.
'''

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	file = open(filename, 'r')
	doc = file.read()
	file.close()
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions


'''
The description text will need to be encoded to numbers before it can be 
presented to the model as in input or compared to the model’s predictions.
The first step in encoding the data is to create a consistent mapping from 
words to unique integer values. Keras provides the Tokenizer class that 
can learn this mapping from the loaded description data.
'''

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

'''
Each description will be split into words. 
The model will be provided one word and the photo and generate the next word. 
Then the first two words of the description will be provided to the model as 
input with the image to generate the next word. 
This is how the model will be trained.

For example, the input sequence “little girl running in field” would be 
split into 6 input-output pairs to train the model:

X1,		X2 (text sequence), 						y (word)
photo	startseq, 									little
photo	startseq, little,							girl
photo	startseq, little, girl, 					running
photo	startseq, little, girl, running, 			in
photo	startseq, little, girl, running, in, 		field
photo	startseq, little, girl, running, in, field, endseq
'''

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
	#X1 : input for photo features
	#X2 : input for text features
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)


# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, 
											max_length, 
											train_descriptions, 
											train_features)



# load val set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
val = load_set(filename)
print('Dataset: %d' % len(val))

# descriptions
val_descriptions = load_clean_descriptions('descriptions.txt', val)
print('Descriptions: val=%d' % len(val_descriptions))

# photo features
val_features = load_photo_features('features.pkl', val)
print('Photos: val=%d' % len(val_features))

# prepare sequences
X1val, X2val, yval = create_sequences(tokenizer, max_length, val_descriptions, val_features)

