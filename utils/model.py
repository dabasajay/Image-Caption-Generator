import numpy as np
# Keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Embedding, concatenate
from keras.preprocessing.sequence import pad_sequences
# To measure BLEU Score
from nltk.translate.bleu_score import corpus_bleu

# Define the CNN model
def CNNModel():
	model = InceptionV3()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	return model

# Define the RNN model
def RNNModel(vocab_size, max_len, rnnConfig):
	embedding_size = rnnConfig['embedding_size']
	# InceptionV3 outputs a 2048 dimensional vector for each image which we'll feed to RNN Model
	image_input = Input(shape=(2048,))
	image_model_1 = Dropout(rnnConfig['dropout'])(image_input)
	image_model = Dense(embedding_size, activation='relu')(image_model_1)

	# Since we are going to predict the next word using the previous words(length of previous words changes with every iteration over the caption), we have to set return_sequences = True.
	caption_input = Input(shape=(max_len,))
	caption_model_1 = Embedding(vocab_size, embedding_size)(caption_input)
	caption_model_2 = Dropout(rnnConfig['dropout'])(caption_model_1)
	caption_model = LSTM(rnnConfig['LSTM_units'])(caption_model_2)

	# Merging the models and creating a softmax classifier
	final_model_1 = concatenate([image_model, caption_model])
	final_model_2 = Dense(rnnConfig['dense_units'], activation='relu')(final_model_1)
	final_model = Dense(vocab_size, activation='softmax')(final_model_2)

	model = Model(inputs=[image_input, caption_input], outputs=final_model)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model

# Map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# Generate a description for an image, given a pre-trained model and a tokenizer to map integer back to word
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = np.argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

def evaluate_model(model, photos, descriptions, tokenizer, max_length):
	actual, predicted = list(), list()
	for key, desc_list in descriptions.items():
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	print('BLEU Scores :')
	print('A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.')
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))