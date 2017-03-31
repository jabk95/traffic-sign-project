Self-Driving Car Engineer Nanodegree
Deep Learning
Project: Build a Traffic Sign Recognition Classifier
In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary.
Note: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to \n", "File -> Download as -> HTML (.html). Include the finished document along with this notebook as your submission.
In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a write up template that can be used to guide the writing process. Completing the code template and writeup template will cover all of the rubric points for this project.
The rubric contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
Note: Code and Markdown cells can be executed using the Shift + Enter keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.
Step 0: Load The Data
In [1]:
# Load pickled data
import pickle
import numpy as np
# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'].astype('f'), train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
Step 1: Dataset Summary & Exploration
The pickled data is a dictionary with 4 key/value pairs:
'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES
Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the pandas shape method might be useful for calculating some of the summary results.
Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas
In [2]:
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43
Include an exploratory visualization of the dataset
Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
The Matplotlib examples and gallery pages are a great resource for doing visualizations in Python.
NOTE: It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.
In [3]:
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline
class data_vis():
    # creates/plots a histogram , and prints the example count of each classes
    def hist(y_train):
        hist, bins = np.histogram(y_train, bins=n_classes)
        width = 1#0.9 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()
        print(np.bincount(y_train))
    
    def image_plot(x,color_scale):
        images_index = np.random.choice(n_train, 3, replace = False)
        images = x[images_index]
        label = y_train[images_index]
        for i, image in enumerate(images_index):
            image = x[image]
            plt.subplot(1, 3, i+1)
            if color_scale == 'gray':
                plt.imshow(image, cmap='gray')
                print(label[i])
            else:    
                plt.imshow(image)
                print(label[i])
plt.show()
In [4]:
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.
%matplotlib inline

# show image of 10 random data points
fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(10):
    index = random.randint(0, len(X_train))
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])

In [5]:
hist_gram = data_vis.hist(y_train) #the dataset is uneven, new data should be generated

[ 180 1980 2010 1260 1770 1650  360 1290 1260 1320 1800 1170 1890 1920  690
  540  360  990 1080  180  300  270  330  450  240 1350  540  210  480  240
  390  690  210  599  360 1080  330  180 1860  270  300  210  210]
First i just plot 10 random pictures. They seem a bit odd, they have a weird shade to them. Then I plot a histogram showing the distributon of the classes. As you can tell they are fairly uneven. Which makes sense because not all signs in the real world are evenyl distributed.
Number of training examples = 34799
Number of testing examples = 12630
Step 2: Design and Test a Model Architecture
Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the German Traffic Sign Dataset.
There are various aspects to consider when thinking about this problem:
Neural network architecture
Play around preprocessing techniques (normalization, rgb to grayscale, etc)
Number of examples per label (some have more than others).
Generate fake data.
Here is an example of a published baseline model on this problem. It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
NOTE: The LeNet-5 implementation shown in the classroom at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!
Pre-process the Data Set (normalization, grayscale, etc.)
Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.
In [6]:
from sklearn.utils import shuffle
from skimage.color import rgb2gray
from sklearn import preprocessing
import cv2
#class for preprocessing data
class pre_proc():
    def shuffle_data(x,y):
        return shuffle(x,y)
    def grayscale(x):
        gray_images = []
        for image in x: 
            
            gray = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
            
            gray_images.append(gray)
        
        return np.array(gray_images)
    #normalize teh grayscale changing /b to +-1 from +- .5 increased my accuracy without the additional data
    def normalize_grayscale(image_data):
        a = -1.0
        b = 1.0
        grayscale_min = 0
        grayscale_max = 255
        return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
       
In [8]:
#shuffle xtrain and ytrain
X_train, y_train = pre_proc.shuffle_data(X_train, y_train)
X_gray = pre_proc.grayscale(X_train)
#normalize X_train
X_gray = pre_proc.normalize_grayscale(X_gray)

print("Updated Image Shape: {}".format(X_gray[0].shape))
Updated Image Shape: (32, 32)
In [9]:
import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.
%matplotlib inline

# show image of 10 random data points
fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(10):
    index = random.randint(0, len(X_gray))
    image = X_gray[index]
    axs[i].axis('off')
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title(y_train[index])

In [10]:
from numpy import newaxis
X_gray = X_gray[...,newaxis]
print("Updated Image Shape: {}".format(X_gray[0].shape))
Updated Image Shape: (32, 32, 1)
In [11]:
X_valid = pre_proc.grayscale(X_valid)
X_valid = pre_proc.normalize_grayscale(X_valid)
X_valid = X_valid[...,newaxis]
print("Updated Image Shape: {}".format(X_valid[0].shape))
Updated Image Shape: (32, 32, 1)
In [ ]:
#histogram equalization; not used
from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
In [15]:
fig, axs = plt.subplots(1,2, figsize=(10, 3))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('normalized')
axs[0].imshow(X_gray[10].squeeze(), cmap='gray')

axs[1].axis('off')
axs[1].set_title('original')
axs[1].imshow(X_train[10].squeeze(), cmap='gray')
Out[15]:
<matplotlib.image.AxesImage at 0x15a5e1d68>

In [16]:
from scipy import ndimage
import random
In [17]:
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec
#generate new data
def augment_brightness_camera_images(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range):
    '''
    This function transforms images to generate new images.
    It rotates, translates and shears image and then adjusts brightness
    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    
    # Brightness 
    

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    img = augment_brightness_camera_images(img)
    
    return img
In [19]:
pics_in_class = np.bincount(y_train)
mean_pics = int(np.mean(pics_in_class))
min_desired = int(mean_pics)
for i in range(len(pics_in_class)):
    
    # Check if less data than the mean
    if pics_in_class[i] < min_desired:
        
        # Count how many additional pictures we want
        new_wanted = min_desired - pics_in_class[i]
        picture = np.where(y_train == i)
        more_X = []
        more_y = []
        
    
      
        # Use the two previous functions to generate data
        for num in range(new_wanted):
            new_img = X_gray[picture][random.randint(0,pics_in_class[i] - 1)]
            new_img = transform_image(new_img, 15, 2, 2)
            new_img = cv2.cvtColor( new_img, cv2.COLOR_RGB2GRAY )
            new_img = new_img[...,newaxis]
            # Rotate images and append new ones to more_X, append the class to more_y
            more_X.append(new_img)
            more_y.append(i)
        
        # Append the pictures generated for each class
        X_gray = np.append(X_gray, np.array(more_X), axis=0)
        y_train = np.append(y_train, np.array(more_y), axis=0)
        
print('Additional data generated.', min_desired, 'pictures.')
Additional data generated. 809 pictures.
In [20]:
fig, axs = plt.subplots(1,2, figsize=(10, 3))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('preprocessed')
axs[0].imshow(X_gray[10].squeeze(), cmap='gray')

axs[1].axis('off')
axs[1].set_title('original')
axs[1].imshow(X_train[10].squeeze(), cmap='gray')
Out[20]:
<matplotlib.image.AxesImage at 0x15a78d668>

In [21]:
#plot the new histogram with updated data
plt.hist(y_train, bins = n_classes)
print(len(y_train))
updated_n_train = len(X_gray)
X_gray,y_train = shuffle(X_gray, y_train)
print("number of training examples =", updated_n_train)
46714
number of training examples = 46714

2.To preprocess the data, I grayscaled it, and then normalized it between -1 and 1. I tried smaller values like -.5, and .5, but it seemed to not do as well as the final solution.
I then created new data by using the functions in cell block 12. These functions took in an image and rotated, translated, sheared, and brightened the images. These were all used in the baseline model so i thought it would be a good starting point to increase the accuracy.
From the histogram above in cell 17, they data is more vast. We now have 46714 examples. In the baseline model provided I believe it used 120 thousand+ examples.
Upon further research I found an interesting method called spatial transformers, which I think a team at Google came up with. From what I understood, implementing that inside the network requires less data examples than the models I tried.
Model Architecture
In [24]:
import tensorflow as tf

EPOCHS = 40
BATCH_SIZE = 100
In [25]:
### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten
In [36]:
#tried to implement the net from the baseline article. 
def neur_net(x):
    mu = 0
    sigma = .1
     #Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    
    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    # SOLUTION: Pooling. Input = 28x28x32. Output = 14x14x32.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x64.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)
    #output 5x5x64
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    
    # SOLUTION: pool again. Input = 5x5x64 output = 2x2x64
    layer2  = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    layer2 = flatten(layer2)
    
    #3rd con layer output 128
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 128), mean = mu, stddev = sigma))
    conv3_B = tf.zeros(128)
    conv3 = tf.nn.conv2d(conv2, conv3_W,  strides=[1,1,1,1], padding='VALID')
    
    #activation of conv3
    conv3 = tf.nn.relu(conv3)
    
    conv3 = tf.nn.dropout(conv3, keep_prob1)
    
    #flatten layer 3; output = 128
    conv3  = flatten(conv3)
    
    conv1 = tf.nn.max_pool(conv1,ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    
    conv1 = flatten(conv1)
    
    conv3 = tf.concat([conv3,layer2, conv1], 1)
    
    #dropout to prevent from overfitting
    conv3 = tf.nn.dropout(conv3, keep_prob2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 672.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1600, 1024), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1024))
    logits   = tf.matmul(conv3, fc1_W) + fc1_b
    #activation
    logits = tf.nn.relu(conv3)
    #final layer
    logits = tf.matmul(logits, tf.Variable(tf.truncated_normal(shape=(1600, 43), mean = mu, stddev = sigma))) + tf.Variable(tf.zeros(43))
    

    return logits
    
    
The network has 3 convolutional layers, one fully connected, and the final output layer. Each layer is fed to the next but also fed directly into the fourth layer: fully connected. I did not experiment with the abtch size. The layers were pooled and flattened and combined to make the input into fully connected layer. I think the imput size for the fully connected layer might be too high. Im not really sure.
5x5 conv
5x5 conv
5x5 conv
dropout
fully connected
dropout
classification
Train, Validate and Test the Model
A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. A high accuracy on the test set but low accuracy on the validation set implies overfitting.
In [27]:
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)
print('done')
done
In [38]:
rate = 0.0009

logits = neur_net(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels = one_hot_y, logits = logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
In [39]:
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob1:1.0, keep_prob2: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
In [40]:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_gray)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_gray, y_train = shuffle(X_gray, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_gray[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob1:0.8, keep_prob2: 0.5})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './trafficnet')
    print("Model saved")
Training...

EPOCH 1 ...
Validation Accuracy = 0.890

EPOCH 2 ...
Validation Accuracy = 0.936

EPOCH 3 ...
Validation Accuracy = 0.956

EPOCH 4 ...
Validation Accuracy = 0.953

EPOCH 5 ...
Validation Accuracy = 0.965

EPOCH 6 ...
Validation Accuracy = 0.964

EPOCH 7 ...
Validation Accuracy = 0.968

EPOCH 8 ...
Validation Accuracy = 0.971

EPOCH 9 ...
Validation Accuracy = 0.968

EPOCH 10 ...
Validation Accuracy = 0.971

EPOCH 11 ...
Validation Accuracy = 0.966

EPOCH 12 ...
Validation Accuracy = 0.969

EPOCH 13 ...
Validation Accuracy = 0.967

EPOCH 14 ...
Validation Accuracy = 0.970

EPOCH 15 ...
Validation Accuracy = 0.892

EPOCH 16 ...
Validation Accuracy = 0.972

EPOCH 17 ...
Validation Accuracy = 0.974

EPOCH 18 ...
Validation Accuracy = 0.965

EPOCH 19 ...
Validation Accuracy = 0.964

EPOCH 20 ...
Validation Accuracy = 0.976

EPOCH 21 ...
Validation Accuracy = 0.966

EPOCH 22 ...
Validation Accuracy = 0.959

EPOCH 23 ...
Validation Accuracy = 0.966

EPOCH 24 ...
Validation Accuracy = 0.971

EPOCH 25 ...
Validation Accuracy = 0.976

EPOCH 26 ...
Validation Accuracy = 0.971

EPOCH 27 ...
Validation Accuracy = 0.970

EPOCH 28 ...
Validation Accuracy = 0.971

EPOCH 29 ...
Validation Accuracy = 0.974

EPOCH 30 ...
Validation Accuracy = 0.960

EPOCH 31 ...
Validation Accuracy = 0.961

EPOCH 32 ...
Validation Accuracy = 0.977

EPOCH 33 ...
Validation Accuracy = 0.969

EPOCH 34 ...
Validation Accuracy = 0.965

EPOCH 35 ...
Validation Accuracy = 0.981

EPOCH 36 ...
Validation Accuracy = 0.976

EPOCH 37 ...
Validation Accuracy = 0.978

EPOCH 38 ...
Validation Accuracy = 0.980

EPOCH 39 ...
Validation Accuracy = 0.978

EPOCH 40 ...
Validation Accuracy = 0.978

Model saved
For Training:
Batch Size: 100
Epochs: 40
Learning rate: .0009
Optimizer: AdamOptimizer
I tried different batch sizes and epochs. I settled on 100 and 40 becuase I got the best result from that however i think 40 might not have been high enough, but it just kind of bumped around between 96 and 97 accuracy until around 35, it continued to increase so I am not really sure how high it could have gone with more epochs.
The adamoptimizer was used in the previous lesson, and upon further research I read nothing but good things about it. I wanted to try a lower learning rate than the original .001, and it helped a little bit.
In [41]:
X_test = pre_proc.grayscale(X_test)
X_test = pre_proc.normalize_grayscale(X_test)
X_test = X_test[..., newaxis]
In [44]:
# Launch the model on the test data
with tf.Session() as sess:
    saver.restore(sess, './trafficnet')
    print('Testing...')
    test_accuracy = sess.run(accuracy_operation, feed_dict={x: X_test, y: y_test, keep_prob1: 1.0, keep_prob2 : 1.0})

print('Test Accuracy: {}'.format(test_accuracy))
Testing...
Test Accuracy: 0.9603325128555298
I started with the LeNet architecture from the previous lessson. This accuracy was not high enough. We were supposed to get the highest accuracy as we could. This architecture is based off of the baseline paper. it consists of three convolutional layers and the final logits layer. its sort of similar to a resnet architecture, based on how they feed layers to the next convolution and also straight to the classifier. its not nearly as deep. They called it multiscale cnn in the paper. I did my best to implement it. They gave a few tips for pooling again after the initial pools of the layers to pass into the fully connected.
I tried histogram equalization because I read a paper written by the winner of the traffic sign classification challenge. They said they used that technique, however it did not help my solution.
I used two dropout layers because I did not want to overfit the data.
i really want to improve upon this though. I did tons of reading into all different methods, and found this all very exciting. i would like to in the future try batch normalization. before this project i had never implemented a cnn on my own before besides in the previous lesson. One i was intrigued by was a cnn with a spatial transformer inside the network. I was not able to fully understand how these work in the short amount of time since i started the project but hope to do further reserch to see if they are actually useful. I believe researchers at Google were the ones to release the paper i read. I also believe in the baseline paper they did pre-training. Not sure if that is true. What i gather is that you train the model on some data first and then train it again on more data. Had i more time and more computational resources, I would have loved to try all different architectures. Looking forward to all critiques of my network.
Validation Accuracy: best was 97.8%
Test Accuracy: 96 %
Step 3: Test a Model on New Images
To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
You may find signnames.csv useful as it contains mappings from the class id (integer) to the actual sign name.
Load and Output the Images
In [137]:
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os
import matplotlib.image as mpimg

newpictures = os.listdir("newpictures/")
newpictures = newpictures[1:6]

new_pics = []
for i in newpictures:
    i = 'newpictures/' + i
    image = mpimg.imread(i)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_pics.append(image)
    plt.imshow(image)
    plt.show()





The images are very blurry, hopefully the preporcessing will make them more readeable.
Predict the Sign Type for Each Image
In [121]:
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

pictures = np.array(new_pics)


print(pictures.shape)
(5, 32, 32, 3)
In [143]:
new_pics = pre_proc.grayscale(new_pics)
new_pics = new_pics[..., newaxis]

# Normalize
new_pictures = pre_proc.normalize_grayscale(new_pics) 

print('Finished preprocessing additional pictures.')


new_image_shape = new_pictures.shape
print("Processed pictures shape =", new_image_shape)
Finished preprocessing additional pictures.
Processed pictures shape = (5, 32, 32, 1)
In [144]:
my_labels = [3, 34, 25, 14, 13]

fig, axs = plt.subplots(1,5, figsize=(15, 5))
fig.subplots_adjust(hspace = .2, wspace=.1)
axs = axs.ravel()
for i in range(5):
    
    image = new_pictures[i]
    
    axs[i].imshow(image.squeeze(), cmap='gray')
    axs[i].set_title(my_labels[i])

Analyze Performance
In [145]:
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


with tf.Session() as sess:
    saver.restore(sess, './trafficnet' )
    my_accuracy = evaluate(new_pictures, my_labels)
    print('testing')
    print("Test Set Accuracy = {:.3f}".format(my_accuracy))
testing
Test Set Accuracy = 0.600
60% accurate on the pictures i got from the internet.
Output Top 5 Softmax Probabilities For Each Image Found on the Web
For each of the new images, print out the model's softmax probabilities to show the certainty of the model's predictions (limit the output to the top 5 probabilities for each image). tf.nn.top_k could prove helpful here.
The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
tf.nn.top_k will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. tk.nn.top_k is used to choose the three classes with the highest probability:
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
Running it through sess.run(tf.nn.top_k(tf.constant(a), k=3)) produces:
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
Looking just at the first row we get [ 0.34763842,  0.24879643,  0.12789202], you can confirm these are the 3 largest probabilities in a. You'll also notice [3, 0, 5] are the corresponding indices.
In [ ]:
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./trafficnet.meta')
    saver.restore(sess, "./trafficnet")
    softmax = sess.run(softmax_logits, feed_dict={x: new_pictures, keep_prob1: 1.0, keep_prob2: 1.0})
    top_k = sess.run(top_k, feed_dict={x: new_pictures, keep_prob1:1.0, keep_prob2: 1.0})
    
      
  
        
    
    
In [159]:
fig, axs = plt.subplots(len(new_pictures),2, figsize=(12, 14))
fig.subplots_adjust(hspace = .4, wspace=.2)
axs = axs.ravel()

for i, image in enumerate(new_pictures):
    axs[2*i].axis('off')
    axs[2*i].imshow(image.squeeze())
    axs[2*i].set_title('added image')
    guess1 = top_k[1][i][0]
    index1 = np.argwhere(y_valid == guess1)[0]
    axs[2*i+1].axis('off')
    axs[2*i+1].imshow(X_valid[index1].squeeze(), cmap='gray')
    axs[2*i+1].set_title('top guess: {} ({:.0f}%)'.format(guess1, 100*top_k[0][i][0]))

In [177]:
for i in range(5):
    print('Image', i, 'predictions', my_top_k[1][i], 'percentage', my_top_k[0][i], 'actual', my_labels[i])
Image 0 predictions [9 5 3] percentage [ 0.95289975  0.03721778  0.00455593] actual 3
Image 1 predictions [34 38 35] percentage [  9.98222768e-01   1.77724240e-03   4.40095397e-16] actual 34
Image 2 predictions [ 0 24 25] percentage [  9.99997973e-01   1.59818251e-06   4.46959319e-07] actual 25
Image 3 predictions [14  3  1] percentage [  1.00000000e+00   5.57395276e-13   1.80260351e-14] actual 14
Image 4 predictions [13  0  1] percentage [ 1.  0.  0.] actual 13
The images it got wrong were the first and third. The third image looks nothing like the guess but it was 46 percent sure it was whaat it actually was. Same with the first image, it got the third guess right. This could be because the image is not as high of quality as it should be but more than likely the model is not as well rounded as i would like it be. It is not as good on 'real world data'. again, looking forward to any comments. I realy want to have more knowledge on this subject. I welcome the crituiques.
Note: Once you have completed all of the code implementations, you need to finalize your work by exporting the IPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run. You can then export the notebook by using the menu above and navigating to \n", "File -> Download as -> HTML (.html). Include the finished document along with this notebook as your submission.
Project Writeup
Once you have completed the code implementation, document your results in a project writeup using this template as a guide. The writeup can be in a markdown or pdf file.
