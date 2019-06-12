from pickle import load
from utils.model import *
from utils.load_data import loadTrainData, loadValData, data_generator
from tensorflow.keras.callbacks import ModelCheckpoint
from config import config, rnnConfig
import random
# Setting random seed for reproducibility of results
random.seed(config['random_seed'])

"""
    *Some simple checking
"""
assert type(config['num_of_epochs']) is int, 'Please provide an integer value for `num_of_epochs` parameter in config.py file'
assert type(config['max_length']) is int, 'Please provide an integer value for `max_length` parameter in config.py file'
assert type(config['batch_size']) is int, 'Please provide an integer value for `batch_size` parameter in config.py file'
assert type(config['beam_search_k']) is int, 'Please provide an integer value for `beam_search_k` parameter in config.py file'
assert type(config['random_seed']) is int, 'Please provide an integer value for `random_seed` parameter in config.py file'
assert type(rnnConfig['embedding_size']) is int, 'Please provide an integer value for `embedding_size` parameter in config.py file'
assert type(rnnConfig['LSTM_units']) is int, 'Please provide an integer value for `LSTM_units` parameter in config.py file'
assert type(rnnConfig['dense_units']) is int, 'Please provide an integer value for `dense_units` parameter in config.py file'
assert type(rnnConfig['dropout']) is float, 'Please provide a float value for `dropout` parameter in config.py file'

"""
	*Load Data
	*X1 : Image features
	*X2 : Text features(Captions)
"""
X1train, X2train, max_length = loadTrainData(config)

X1val, X2val = loadValData(config)

"""
	*Load the tokenizer
"""
tokenizer = load(open(config['tokenizer_path'], 'rb'))
vocab_size = len(tokenizer.word_index) + 1

"""
	*Now that we have the image features from CNN model, we need to feed them to a RNN Model.
	*Define the RNN model
"""
# model = RNNModel(vocab_size, max_length, rnnConfig, config['model_type'])
model = AlternativeRNNModel(vocab_size, max_length, rnnConfig, config['model_type'])
print('RNN Model (Decoder) Summary : ')
print(model.summary())

"""
    *Train the model save after each epoch
"""
num_of_epochs = config['num_of_epochs']
batch_size = config['batch_size']
steps_train = len(X2train)//batch_size
if len(X2train)%batch_size!=0:
    steps_train = steps_train+1
steps_val = len(X2val)//batch_size
if len(X2val)%batch_size!=0:
    steps_val = steps_val+1
model_save_path = config['model_data_path']+"model_"+str(config['model_type'])+"_epoch-{epoch:02d}_train_loss-{loss:.4f}_val_loss-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

print('steps_train: {}, steps_val: {}'.format(steps_train,steps_val))
print('Batch Size: {}'.format(batch_size))
print('Total Number of Epochs = {}'.format(num_of_epochs))

# Shuffle train data
ids_train = list(X2train.keys())
random.shuffle(ids_train)
X2train_shuffled = {_id: X2train[_id] for _id in ids_train}
X2train = X2train_shuffled

# Create the train data generator
# returns [[img_features, text_features], out_word]
generator_train = data_generator(X1train, X2train, tokenizer, max_length, batch_size, config['random_seed'])
# Create the validation data generator
# returns [[img_features, text_features], out_word]
generator_val = data_generator(X1val, X2val, tokenizer, max_length, batch_size, config['random_seed'])

# Fit for one epoch
model.fit_generator(generator_train,
            epochs=num_of_epochs,
            steps_per_epoch=steps_train,
            validation_data=generator_val,
            validation_steps=steps_val,
            callbacks=callbacks,
            verbose=1)

"""
	*Evaluate the model on validation data and ouput BLEU score
"""
print('Model trained successfully. Running model on validation set for calculating BLEU score using BEAM search with k={}'.format(config['beam_search_k']))
evaluate_model_beam_search(model, X1val, X2val, tokenizer, max_length, beam_index=config['beam_search_k'])