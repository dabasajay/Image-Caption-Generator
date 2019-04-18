<h2>Image Caption Generator</h2>

A Neural Network to generate captions for an image.

<p align="center">
  <strong>Examples</strong>
</p>

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*6BFOIdSHlk24Z3DFEakvnQ.png" width="85%" title="Example of Image Captioning" alt="Example of Image Captioning">
</p>

<p align="center">
	Credits : <a href="https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2">Towardsdatascience</a>
</p>

<h4>Requirements</h4>

Recommended System Requirements to train model.

<ul type="square">
	<li>A good CPU and a GPU with atleast 4gb memory</li>
	<li>Atleast 8gb of RAM</li>
	<li>Active internet connection so that keras can download inceptionv3 model weights</li>
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
	<a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/">Download Link Credits: Jason Brownlee</a>
</ul>

<strong>Important:</strong> After downloading the dataset, put the reqired files in train_val_data folder

<h4>Procedure to Train Model</h4>
<ol>
	<li>Clone the repository to preserve directory structure</li>
	<li>Put the required dataset files in train_val_data Folder (files mentioned in readme there)</li>
	<li>Review config.py for paths and other configurations</li>
	<li>Run train_val.py</li>
</ol>

<h4>Procedure to Test on new images (After training only)</h4>
<ol>
	<li>Clone the repository to preserve directory structure</li>
	<li>Train the model to generate required files in model_data folder</li>
	<li>Put the test image in test_data folder</li>
	<li>Review config.py for paths and other configurations</li>
	<li>Run test.py</li>
</ol>

<h4>References</h4>

<ul type="square">
	<li><a href="https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf">Show and Tell: A Neural Image Caption Generator</a> - Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan</li>
	<li><a href="https://arxiv.org/abs/1703.09137">Where to put the Image in an Image Caption Generator</a> - Marc Tanti, Albert Gatt, Kenneth P. Camilleri</li>
	<li><a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/"></a>How to Develop a Deep Learning Photo Caption Generator from Scratch</li>
</ul>