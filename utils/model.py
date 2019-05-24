import numpy as np
# Keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Embedding, concatenate
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
# To measure BLEU Score
from nltk.translate.bleu_score import corpus_bleu

"""
	*Define the CNN model
"""
def CNNModel():
	model = InceptionV3()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	return model

"""
	*Define the RNN model
"""
def RNNModel(vocab_size, max_len, rnnConfig):
	embedding_size = rnnConfig['embedding_size']
	# InceptionV3 outputs a 2048 dimensional vector for each image which we'll feed to RNN Model
	image_input = Input(shape=(2048,))
	image_model_1 = Dropout(rnnConfig['dropout'])(image_input)
	image_model = Dense(embedding_size, activation='relu')(image_model_1)

	# Since we are going to predict the next word using the previous words
	# (length of previous words changes with every iteration over the caption), we have to set return_sequences = True.
	caption_input = Input(shape=(max_len,))
	# mask_zero: We zero pad inputs to the same length, the zero mask ignores those inputs. E.g. it is an efficiency.
	caption_model_1 = Embedding(vocab_size, embedding_size, mask_zero=True)(caption_input)
	caption_model_2 = Dropout(rnnConfig['dropout'])(caption_model_1)
	caption_model = LSTM(rnnConfig['LSTM_units'])(caption_model_2)

	# Merging the models and creating a softmax classifier
	final_model_1 = concatenate([image_model, caption_model])
	final_model_2 = Dense(rnnConfig['dense_units'], activation='relu')(final_model_1)
	final_model = Dense(vocab_size, activation='softmax')(final_model_2)

	model = Model(inputs=[image_input, caption_input], outputs=final_model)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model

"""
	*Map an integer to a word
"""
def int_to_word(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

"""
	*Generate a caption for an image, given a pre-trained model and a tokenizer to map integer back to word
"""
def generate_caption(model, tokenizer, image, max_length):
	# Seed the generation process
	in_text = 'startseq'
	# Iterate over the whole length of the sequence
	for _ in range(max_length):
		# Integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# Pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# Predict next word
		# The model will output a prediction, which will be a probability distribution over all words in the vocabulary.
		yhat = model.predict([image,sequence], verbose=0)
		# The output vector representins a probability distribution where maximum probability is the predicted word position
		# Take output class with maximum probability and convert to integer
		yhat = np.argmax(yhat)
		# Map integer back to word
		word = int_to_word(yhat, tokenizer)
		# Stop if we cannot map the word
		if word is None:
			break
		# Append as input for generating the next word
		in_text += ' ' + word
		# Stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

"""
	*Evaluate the model on BLEU Score
"""
def evaluate_model(model, images, captions, tokenizer, max_length):
	actual, predicted = list(), list()
	for image_id, caption_list in tqdm(captions.items()):
		yhat = generate_caption(model, tokenizer, images[image_id], max_length)
		ground_truth = [caption.split() for caption in caption_list]
		actual.append(ground_truth)
		predicted.append(yhat.split())
	print('BLEU Scores :')
	print('A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.')
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))