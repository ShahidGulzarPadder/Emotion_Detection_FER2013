# Emotion_Detection_FER2013
In this repository I have built and trained a multimodal deep neural network for emotion detection using tf.keras/pytorch. Here, I worked with the FER2013 dataset, which contains more than 28000 images. The images are automatically gathered, so there can be mislabeled or bad quality samples as well. Every image has a single label from the following list: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
There is no constraint, you can preprocess the data as you wish, use any DNN (or Transformer architecture) for training, etc. The expected solution should contain data loading and visualization, preprocessing, model definition, training and evaluation.

# Example Pipeline
Here, there is an example pipeline with such details to help you out:
* Download the FER2013 dataset, it is available in csv format at the following link: nipg1.inf.elte.hu:8765/fer2013.csv (if the file is unreachable, write an email)
* You have to solve a classification task.
* Preprocess the data and visualize samples
  * gather the images, separate the train-valid-test subsets (Training, PublicTest, PrivateTest)
  * resize the images: original size is 48x48, networks usually expect e.g. 224x224x3 images. Do the resizing part during data augmentation, otherwise you run out of Colab resources.
  * all images are grayscale with a single value. duplicate the channel dimension if necessary to transform (H,W) to (H,W,3)
  * plot label histogram
* Define train-valid-test dataloaders
  * augment and normalize the input images. (standardize the inputs using the dataset mean/std values, or a preprocess function expected by a pretrained network) - you can use a resize transform to get 240x240x3 images then apply a 224x224x3 randomcrop. 
* Define and train a model
  * define a model (e.g. VGG11 or ResNet18, etc)
  * define the loss function, optimizer
  * use early stopping and a learning rate scheduler
  * train the model using the training and validation subsets
  * plot the training/validation curve
* Evaluation
  * evaluate the model on the test set
  * print/plot classification results and confusion matrix
  * visualize some misclassified samples

At this point, you have the results for the imbalanced case. 

* Balancing
  * Modify it slighthly in the dataloader (resampling) or at loss definition (weighting).
* Compare the classification metrics and confusion matrices

# Use GPU
Runtime -> Change runtime type

At Hardware accelerator select GPU then save it.


# Useful shortcuts
* Run selected cell: *Ctrl + Enter*
* Insert cell below: *Ctrl + M B*
* Insert cell above: *Ctrl + M A*
* Convert to text: *Ctrl + M M*
* Split at cursor: *Ctrl + M -*
* Autocomplete: *Ctrl + Space* or *Tab*
* Move selected cells up: *Ctrl + M J*
* Move selected cells down: *Ctrl + M K*
* Delete selected cells: *Ctrl + M D*
