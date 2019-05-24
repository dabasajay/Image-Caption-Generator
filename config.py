# All paths are relative to train_val.py file
config = {
	'images_path': 'train_val_data/Flicker8k_Dataset/', #Make sure you put that last slash(/)
	'train_data_path': 'train_val_data/Flickr_8k.trainImages.txt',
	'val_data_path': 'train_val_data/Flickr_8k.devImages.txt',
	'captions_path': 'train_val_data/Flickr8k.token.txt',
	'tokenizer_path': 'model_data/tokenizer.pkl',
	'model_data_path': 'model_data/', #Make sure you put that last slash(/)
	'model_load_path': 'model_data/model_epoch-01_train_loss-4.5119_val_loss-3.9564.hdf5',
	'num_of_epochs': 15,
	'max_length': 34, #This is set during training of model
	'test_data_path': 'test_data/', #Make sure you put that last slash(/)
	'animation': False,
	'anim_time_int': 1500 #Animation time interval between frames in milliseconds
}

rnnConfig = {
	'embedding_size': 300,
	'LSTM_units': 256,
	'dense_units': 256,
	'dropout': 0.5
}