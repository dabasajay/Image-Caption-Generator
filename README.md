## Image Caption Generator

[![GitHub issues](https://img.shields.io/github/issues/dabasajay/Image-Caption-Generator.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/issues)
[![GitHub forks](https://img.shields.io/github/forks/dabasajay/Image-Caption-Generator.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/network)
[![GitHub stars](https://img.shields.io/github/stars/dabasajay/Image-Caption-Generator.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/stargazers)
[![GitHub license](https://img.shields.io/github/license/dabasajay/Image-Caption-Generator.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/blob/master/LICENSE)

A neural network to generate captions for an image.

<p align="center">
  <strong>Examples</strong>
</p>

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*6BFOIdSHlk24Z3DFEakvnQ.png" width="85%" title="Example of Image Captioning" alt="Example of Image Captioning">
</p>

<p align="center">
	Image Credits : <a href="https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2">Towardsdatascience</a>
</p>

## Requirements

Recommended System Requirements to train model.

<ul type="square">
	<li>A good CPU and a GPU with atleast 4gb memory</li>
	<li>Atleast 8gb of RAM</li>
	<li>Active internet connection so that keras can download inceptionv3/vgg16 model weights</li>
</ul>

Required Libraries for Python along with their version numbers used while making & testing of this project

<ul type="square">
	<li>Python - 3.6.7</li>
	<li>Numpy - 1.16.2</li>
	<li>Keras - 2.2.4</li>
	<li>nltk - 3.2.5</li>
	<li>PIL - 4.3.0</li>
	<li>Matplotlib - 3.0.3</li>
	<li>tqdm - 4.28.1</li>
</ul>

<strong>Flickr8k Dataset:</strong> <a href="https://forms.illinois.edu/sec/1713398">Dataset Request Form</a>

<strong>UPDATE (April/2019):</strong> The official site seems to have been taken down (although the form still works). Here are some direct download links:
<ul type="square">
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip">Flickr8k_Dataset</a></li>
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip">Flickr8k_text</a></li>
	Download Link Credits:<a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/"> Jason Brownlee</a>
</ul>

<strong>Important:</strong> After downloading the dataset, put the reqired files in train_val_data folder

## Procedure to Train Model

1. Clone the repository to preserve directory structure
2. Put the required dataset files in train_val_data Folder (files mentioned in readme there)
3. Review `config.py` for paths and other configurations (explained below)
4. Run `train_val.py`

## Procedure to Test on new images

1. Clone the repository to preserve directory structure
2. Train the model to generate required files in model_data folder
3. Put the test image in test_data folder
4. Review `config.py` for paths and other configurations (explained below)
5. Run `test.py`

## Configurations (config.py)

#### config

1. **`images_path`** :- Folder path containing flickr dataset images
2. `train_data_path` :- .txt file path containing images ids for training
3. `val_data_path` :- .txt file path containing imgage ids for validation
4. `captions_path` :- .txt file path containing captions
5. `tokenizer_path` :- path for saving tokenizer
6. `model_data_path` :- path for saving files related to model
7. **`model_load_path`** :- path for loading trained model
8. **`num_of_epochs`** :- Number of epochs
9. **`max_length`** :- Maximum length of captions. This is set manually after training of model and required for test.py
10. **`batch_size`** :- Batch size for training (larger will consume more GPU & CPU memory)
11. `test_data_path` :- Folder path containing images for testing/inference
12. **`model_type`** :- CNN Model type to use -> inceptionv3 or vgg16
13. **`random_seed`** :- Random seed for reproducibility of results

#### rnnConfig

1. **`embedding_size`** :- Embedding size used in Decoder(RNN) Model
2. **`LSTM_units`** :- Number of LSTM units in Decoder(RNN) Model
3. **`dense_units`** :- Number of Dense units in Decoder(RNN) Model
4. **`dropout`** :- Dropout probability used in Dropout layer in Decoder(RNN) Model

## TODO

- [X] Support for VGG16 Model. Uses InceptionV3 Model by default

- [X] Support for 2 architectures of RNN (Decoder) Model

- [X] Support for batch processing in data generator with shuffling

- [ ] Implement BEAM Search

- [ ] Implement Attention

- [ ] Support for pre-trained word vectors like word2vec, GloVe etc.

## References

<ul type="square">
	<li><a href="https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf">Show and Tell: A Neural Image Caption Generator</a> - Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan</li>
	<li><a href="https://arxiv.org/abs/1703.09137">Where to put the Image in an Image Caption Generator</a> - Marc Tanti, Albert Gatt, Kenneth P. Camilleri</li>
	<li><a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/">How to Develop a Deep Learning Photo Caption Generator from Scratch</a></li>
</ul>
