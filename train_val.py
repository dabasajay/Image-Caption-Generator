from utils.model import *



# define the model
model = define_model(vocab_size, max_length)

# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
								verbose=1, save_best_only=True, 
								mode='min')

# fit model
model.fit([X1train, X2train], ytrain, 
			epochs=20, verbose=2, callbacks=[checkpoint], 
			validation_data=([X1val, X2val], yval))