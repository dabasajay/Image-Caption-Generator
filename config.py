# All paths are relative to train_val.py file
config = {
	'images_path': 'train_val_data/Flicker8k_Dataset/', #Make sure you put that last slash(/)
	'train_data_path': 'train_val_data/Flickr_8k.trainImages.txt',
	'val_data_path': 'train_val_data/Flickr_8k.devImages.txt',
	'captions_path': 'train_val_data/Flickr8k.token.txt',
	'tokenizer_path': 'model_data/tokenizer.pkl',
	'model_data_path': 'model_data/', #Make sure you put that last slash(/)
	'model_load_path': 'model_data/model_vgg16_epoch-05_train_loss-3.4372_val_loss-3.8633.hdf5',
	'num_of_epochs': 5,
	'max_length': 34, #This is set during training of model
	'test_data_path': 'test_data/', #Make sure you put that last slash(/)
	'model_type': 'inceptionv3', # inceptionv3 or vgg16
	'random_seed': 1035
}

rnnConfig = {
	'embedding_size': 128,
	'LSTM_units': 128,
	'dense_units': 128,
	'dropout': 0.3
}