from pickle import load
from utils.model import *
from utils.load_data import loadTrainData, loadValData, data_generator
from tensorflow.keras.callbacks import ModelCheckpoint
from config import config, rnnConfig
import random
random.seed(1035)

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
model = RNNModel(vocab_size, max_length, rnnConfig)
print('RNN Model (Decoder) Summary : ')
print(model.summary())

"""
    *Train the model, run epochs manually and save after each epoch
"""
num_of_epochs = config['num_of_epochs']

if(config['animation']):
    def selectRandom(dict):
        randomNum = random.randint(0,len(dict)-1)
        i=0
        for key,_ in dict.items():
            if(i==randomNum):
                return key
            i = i+1
    selected_image_id_train = selectRandom(X1train)
    selected_image_id_val = selectRandom(X1val)
    selected_image_train = X1train[selected_image_id_train]
    selected_image_val = X1val[selected_image_id_val]
    captions_for_image_train = []
    captions_for_image_val = []

steps_train = len(X2train)
steps_val = len(X2val)
model_save_path = config['model_data_path']+"model_epoch-{epoch:02d}_train_loss-{loss:.4f}_val_loss-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]
print('Total Number of Epochs = ',num_of_epochs)
for i in range(num_of_epochs):
    print('Epoch : ',i+1)
    # Shuffle train data
    ids_train = list(X2train.keys())
    random.shuffle(ids_train)
    X2train_shuffled = {_id: X2train[_id] for _id in ids_train}
    X2train = X2train_shuffled
    # Shuffle validation data though it's not necessary since it's never used for training, just for validation
    ids_val = list(X2val.keys())
    random.shuffle(ids_val)
    X2val_shuffled = {_id: X2val[_id] for _id in ids_val}
    X2val = X2val_shuffled
    # Create the train data generator
    # returns [[img_features, text_features], out_word]
    generator_train = data_generator(X1train, X2train, tokenizer, max_length)
    # Create the validation data generator
    # returns [[img_features, text_features], out_word]
    generator_val = data_generator(X1val, X2val, tokenizer, max_length)
    # Fit for one epoch
    model.fit_generator(generator_train,
                epochs=1,
                steps_per_epoch=steps_train,
                validation_data=generator_val,
                validation_steps=steps_val,
                callbacks=callbacks,
                verbose=1)
    if(config['animation']):
        captions_for_image_train.append(generate_caption(model,tokenizer,selected_image_train,max_length))
        captions_for_image_val.append(generate_caption(model,tokenizer,selected_image_val,max_length))

"""
    *Generate GIF
"""
if(config['animation']):
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    trainOut=True
    print('Generating GIFs now')
    for selected_image_id in [selected_image_id_train,selected_image_id_val]:
        pil_im = Image.open(config['images_path']+selected_image_id+'.jpg', 'r')
        fig, ax = plt.subplots(figsize=(12, 12))
        _ = fig.set_tight_layout(True)
        _ = ax.get_yaxis().set_visible(False)
        _ = ax.imshow(np.asarray(pil_im), interpolation='nearest')
        if(trainOut):
            captions_for_image = captions_for_image_train
        else:
            captions_for_image = captions_for_image_val
        def update(i):
            # remove startseq and endseq
            caption = '\nCaption: ' + captions_for_image[i].split()[1].capitalize()
            for x in captions_for_image[i].split()[2:len(captions_for_image[i].split())-1]:
                caption = caption + ' ' + x
            caption += '.'
            _ = ax.set_xlabel('Epoch: '+str(i+1)+caption,
                fontdict={'fontsize': '20','fontweight' : '50'})
            return ax
        # FuncAnimation will call the 'update' function for each frame; here
        # animating over num_of_epochs frames, with an interval of 1000ms between frames.
        anim = FuncAnimation(fig, update, frames=np.arange(0, len(captions_for_image)), interval=config['anim_time_int'])
        if(trainOut):
            _ = anim.save('duringTraining.gif', dpi=80, writer='imagemagick')
            trainOut = False
        else:
            _ = anim.save('duringValidation.gif', dpi=80, writer='imagemagick')
    print('GIFs Generated!')

"""
	*Evaluate the model on validation data and ouput BLEU score
"""
print('Model trained successfully. Running model on validation set now')
evaluate_model(model, X1val, X2val, tokenizer, max_length)