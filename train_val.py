from pickle import load
from utils.model import *
from utils.load_data import *
from config import config, rnnConfig

"""
	*Load Data
	*X1 : Image features
	*X2 : Text features
"""
# If you've already processed the data once, you can avoid doing it again by setting preprocessDataReady=True
X1train, X2train, max_length = loadTrainData(config,preprocessDataReady=False)

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
print('Model Summary : ')
print(model.summary())

"""
    *Train the model, run epochs manually and save after each epoch
"""
num_of_epochs = config['num_of_epochs']

if(config['animation']):
    from random import randint
    def selectRandom(dict):
        randomNum = randint(0,len(dict)-1)
        i=0
        for key,item in dict.items():
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
print('Total Numer of Epochs = ',num_of_epochs)
for i in range(num_of_epochs):
    print('Epoch : ',i+1)
    # create the train data generator
    # returns [[img_features, text_features], out_word]
    generator_train = data_generator(X1train, X2train, tokenizer, max_length)
    # create the val data generator
    # returns [[img_features, text_features], out_word]
    generator_val = data_generator(X1val, X2val, tokenizer, max_length)
    # fit for one epoch
    model.fit_generator(generator_train,
                epochs=1,
                steps_per_epoch=steps_train,
                validation_data=generator_val,
                validation_steps=steps_val,
                verbose=1)
    if(config['animation']):
        captions_for_image_train.append(generate_desc(model,tokenizer,selected_image_train,max_length))
        captions_for_image_val.append(generate_desc(model,tokenizer,selected_image_val,max_length))
    # save model
    model.save(config['model_save_path']+ 'model_epoch_' + str(i) + '.h5')

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
        anim = FuncAnimation(fig, update, frames=np.arange(0, config['num_of_epochs']), interval=config['anim_time_int'])
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