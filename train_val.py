from pickle import load
from utils.model import *
from utils.load_data import *

# Load Data
# X1 : image features
# X2 : text features
X1train, X2train, max_length 	=	loadTrainData(path = 'Flickr8k_text/Flickr_8k.trainImages.txt',preprocessDataReady=False)

X1val, X2val = loadValData(path = 'Flickr_8k.devImages.txt')

# load the tokenizer
tokenizer_path = 'tokenizer.pkl'
tokenizer = load(open(tokenizer_path, 'rb'))
vocab_size = len(tokenizer.word_index) + 1

# prints 34
print('Max Length : ',max_length) 

# We already have the image features from CNN model so we only need to define the RNN model now.
# define the RNN model
model = defineRNNmodel(vocab_size, max_length)

# train the model, run epochs manually and save after each epoch
epochs = 10
steps_train = len(X2train)
steps_val = len(X2val)
for i in range(epochs):
    # create the train data generator
    generator_train = data_generator(X1train, X2train, tokenizer, max_length)
    # create the val data generator
    generator_val = data_generator(X1val, X2val, tokenizer, max_length)
    # fit for one epoch
    model.fit_generator(generator_train, epochs=1, steps_per_epoch=steps_train, 
                        verbose=1, validation_data=generator_val, validation_steps=steps_val)
    # save model
    model.save('model_' + str(i) + '.h5')

# Evaluate the model on validation data and ouput BLEU score
# evaluate_model(model, X1val, X2val, tokenizer, max_length)