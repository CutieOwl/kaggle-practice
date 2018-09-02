
# coding: utf-8

# # A model of Salt Identification with Keras ConvNet
# Transfer learning will probably be harder with the wonky pictures of salt identification.

# In[1]:


# % matplotlib inline
import datetime as dt

# Import plotting for visualization
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 16
#from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import numpy as np
import os
import pandas as pd
import pydot
import random, math
import itertools
import tensorflow as tf
from keras.applications import xception
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Concatenate
from keras.layers.core import Lambda
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, load_model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm, tqdm_notebook, tnrange
from itertools import chain
import sys
sys.path.append("/utils/DLWorkspace-Utils/keras-multiprocess-image-data-generator/tools")
import image as T
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix # for the confusion matrix

import keras.backend as K
K.set_image_data_format('channels_last')

import warnings
import cv2
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


# ## First, let's set up our optimizer so we can choose which one later on.

# In[2]:


optimizer_collections = {
    "adadelta" : Adadelta(), 
    "nadam" : Nadam(), 
    "rmsprop": RMSprop(), 
    "adam": Adam(), 
    "adagrad": Adagrad(), 
    "adamax": Adamax(), 
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, help='Batch size', type=int)
parser.add_argument('--nb_epochs', default=30, help='Number of Epochs', type=int)
parser.add_argument('--optimizer', default='adam', help='Optimizer', type=str)
parser.add_argument('--split', default=None, help='Which data set to use', type=int)
parser.add_argument('--decay', default=6e-4, help='Rate decay', type=float)
parser.add_argument('--gpus', default=1, help='Number of GPU', type=int)
parser.add_argument('--drops_epochs', default=0, help='Epochs which rate drop by x10', type=float)
parser.add_argument('--lr', default=3e-3, help='Learning Rate', type=float)
parser.add_argument('--epsilon', default=1e-6, help='Optimizer Epsilon', type=float)
parser.add_argument('--rho', default=0.95, help='Optimizer Rho', type=float)
parser.add_argument('--dropout', default=0, help='Dropout', type=float)
parser.add_argument('--reducelr', default=0, help='Reduce learning rate? 0 or 1', type=int)
parser.add_argument('--pooltype', default='avg', help='avg or max pool', type=str)
parser.add_argument('--upsample', default=0, help='1? upsample:conv2dtranspose', type=int)

#args = parser.parse_args("--optimizer adam --dropout 0.5 --reducelr 1".split())
args = parser.parse_args()

print( args )
BATCH_SIZE = 8
if args.batch_size:
    BATCH_SIZE = args.batch_size

num_gpu = args.gpus
# Number of training epochs
EPOCHS = args.nb_epochs
# data to use. 
split = args.split

DO = args.dropout

LR = args.lr
DECAY = args.decay
EPS = args.epsilon
RHO = args.rho

if args.optimizer.startswith("sgd"):
    optimizer = args.optimizer
    opt = SGD(lr = LR, decay=DECAY, momentum=0.9, nesterov=True)
elif args.optimizer.startswith("adam"):
    optimizer = args.optimizer
    opt = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=EPS, decay=DECAY)
elif args.optimizer.startswith("adadelta"):
    optimizer = args.optimizer
    opt = Adadelta(lr=LR, rho=RHO, epsilon=EPS, decay=DECAY)  
elif args.optimizer.startswith("rmsprop"):
    optimizer = args.optimizer
    opt = RMSprop(lr=LR, rho=RHO, epsilon=EPS, decay=DECAY)
else:
    optimizer = args.optimizer
    opt = optimizer_collections[args.optimizer]


# ### This is a tee object to help write log output.

# In[3]:


class Tee(object):
    def __init__(self, name):
        self.file = open(name, "w")
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()


# ## Reading in the dataset
# First, we locate our data at the correct directories.

# In[4]:


data_dir = '/data/kaggle/competitions/tgs-salt-identification-challenge/'
cur_dir = '/work/kaggle-practice/tgs_salt_identification/'
result_dir = os.path.join(cur_dir, 'result')
train_dir = os.path.join(data_dir, 'train')
train_dir_img = os.path.join(train_dir, 'images')
train_dir_mask = os.path.join(train_dir, 'masks')
print(train_dir)
test_dir = os.path.join(data_dir, 'test')
test_dir = os.path.join(test_dir, 'images')
#sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
IMAGE_SIZE = 128 # pixel height and width of each image--actual size of the images is 101


# Here's a little piece of code if we wanted to test out different lrs and decays
# ```python
# if os.path.isdir(os.path.join(result_dir, "tgs-salt_%s_batch%s_epoch%s_lr%s_decay%s" % (optimizer, batch_size, fit_epochs, lr_top, args.decay))) == False:
#     os.mkdir(os.path.join(result_dir, "tgs-salt_%s_batch%s_epoch%s_lr%s_decay%s" % (optimizer, batch_size, fit_epochs, lr_top, args.decay)))
#     
# model_dir = os.path.join(result_dir, "tgs-salt_%s_batch%s_epoch%s_lr%s_decay%s" % (optimizer, batch_size, fit_epochs, lr_top, args.decay))
# ```
# 
# And this was the naming before I split it up
# ```python
# "8.14-tgs-salt_%s_batch%s_epoch%s" % (optimizer, BATCH_SIZE, EPOCHS)
# ```

# In[5]:


# Set up location of output
prefix = "8.27"
info = ("_%s_batch%s_epoch%s_dropout%s_pool%s_upsample%s" % (optimizer, BATCH_SIZE, EPOCHS, DO, args.pooltype, args.upsample))
if os.path.isdir(os.path.join(result_dir, prefix + "-tgs-salt"  + info)) == False:
    os.mkdir(os.path.join(result_dir, prefix + "-tgs-salt"  + info))
model_dir = os.path.join(result_dir, prefix + "-tgs-salt"  + info)
if os.path.exists(os.path.join(model_dir, "training_log")) == True:
    os.remove(os.path.join(model_dir, "training_log"))
log_file = os.path.join(model_dir, "training_log")
print(prefix + "-tgs-salt"  + info)
sys.stdout = Tee(log_file)


# ### Reading in data from files.

# In[6]:


train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col="id", usecols=[0])
depths_df = pd.read_csv(os.path.join(data_dir, "depths.csv"), index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]


# Organize data structures.

# In[7]:


MAX_DEPTH = max(train_df["z"])
MIN_DEPTH = min(train_df["z"])
print("Max: %s, Min: %s" % (MAX_DEPTH, MIN_DEPTH))
train_df["images"] = [np.array(load_img(os.path.join(train_dir_img, '%s.png' % idx), grayscale=True)) / 255 
                      for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img(os.path.join(train_dir_mask, '%s.png' % idx), grayscale=True)) / 255 
                     for idx in tqdm_notebook(train_df.index)]
train_df["depth"] = [((np.ones_like(train_df.loc[i]["images"]) * train_df.loc[i]["z"]) - MIN_DEPTH)/(MAX_DEPTH - MIN_DEPTH)
                     for i in tqdm_notebook(train_df.index)]
print(train_df["depth"][0].shape)
train_df["images_d"] = [np.dstack((train_df["images"][i], train_df["depth"][i])) for i in tqdm_notebook(train_df.index)]
print(train_df["images_d"][0].shape)
train_df.head()


# ### Let's check the distribution of the data.
# Depth first!

# In[8]:


'''
_ = sns.distplot(train_df.z, label="Train")
_ = sns.distplot(test_df.z, label="Test")
_ = plt.legend()
_ = plt.title("Depth distribution")
'''


# Mask coverage...
# Notice that most of the pictures don't have a lot of mask coverage--the data isn't even.

# In[9]:


def coverage(mask):
    """Compute salt mask coverage"""
    return np.sum(mask) / (mask.shape[0]*mask.shape[1])

def coverage_class(mask):
    """Compute salt mask coverage class"""
    return (coverage(mask) * 100 //10).astype(np.int8)

'''
_ = sns.distplot(train_df.masks.map(coverage_class), label="Train", kde=False)
_ = plt.legend()
_ = plt.title("Coverage distribution")
'''


# And let's visualize the training data!

# In[10]:


def plot_imgs_masks(imgs, masks, preds_valid=None, thres=None, grid_width=10, zoom=1.5):
    """Visualize seismic images with their salt area mask(green) and optionally salt area prediction(pink). 
    The prediction mask can be either in probability-mask or binary-mask form(based on threshold)
    """
    grid_height = 1 + (len(imgs)-1) // grid_width
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width*zoom, grid_height*zoom))
    axes = axs.ravel()
    
    for i, img in enumerate(imgs):
        mask = masks[i]
        depth = img[0, 0, 1]
        
        
        ax = axes[i] #//grid_width, i%grid_width]
        _ = ax.imshow(img[..., 0], cmap="Greys")
        _ = ax.imshow(mask, alpha=0.3, cmap="Greens")
        
        if preds_valid is not None:
            if thres is not None:
                pred = np.array(np.round(preds_valid[i] > thres), dtype=np.float32)
            else:
                pred = preds_valid[i]
            _ = ax.imshow(pred, alpha=0.3, cmap="OrRd")
        
        _ = ax.text(2, img.shape[0]-2, depth * MAX_DEPTH//1, color="k")
        _ = ax.text(img.shape[0]-2, 2, round(coverage(mask), 2), color="k", ha="right", va="top")
        _ = ax.text(2, 2, coverage_class(mask), color="k", ha="left", va="top")
        
        _ = ax.set_yticklabels([])
        _ = ax.set_xticklabels([])
        _ = plt.axis('off')
    plt.suptitle("Green: Salt area mask \nTop-left: coverage class, top-right: salt coverage, bottom-left: depth", y=1+.5/grid_height, fontsize=20)
    plt.tight_layout();
    
# show 40 images with their masks overlaid in green
'''
plot_imgs_masks(train_df.iloc[:40].images_d, train_df.iloc[:40].masks)
'''


# ### Split the train and validation sets:

# In[11]:


def upsample(img, img_size_target=IMAGE_SIZE):
    """Resize image to target"""
    img_size = img.shape[0]
    if img_size == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

def downsample(img, img_size_orig=101):
    """Resize image to original"""
    img_size = img.shape[0]
    if img_size == img_size_orig:
        return img
    return resize(img, (img_size_orig, img_size_orig), mode='constant', preserve_range=True)


# In[12]:


ids_train, ids_valid, x_train, x_valid, y_train, y_valid, depth_train, depth_valid = train_test_split(
    train_df.index.values,
    np.array(train_df.images_d.map(upsample).tolist()).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 2), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1), 
    train_df.z.values,
    test_size=0.2, 
    stratify=train_df.masks.map(coverage_class), 
    random_state=1)

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)


# ### Image augmentation time!
# According to Frank on Kaggle, shifts, zooming out, and vertical flip are not useful, so I only have zooming in and horizontal flip.

# In[13]:


NB_BATCHES = 300

# The images and masks need to modified the same way, so their ImageDataGenerators are the same.
image_datagen = ImageDataGenerator(zoom_range=[.9, 1],
                                   horizontal_flip=True,)
mask_datagen = ImageDataGenerator(zoom_range=[.9, 1],
                                  horizontal_flip=True,)

# Provide the same seed and keyword arguments to the fit and flow methods
image_datagen.fit(x_train, seed=1)
mask_datagen.fit(y_train, seed=1)

image_generator = image_datagen.flow(
    x_train,
    batch_size=BATCH_SIZE,
    seed=1)

mask_generator = mask_datagen.flow(
    y_train,
    batch_size=BATCH_SIZE,
    seed=1)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)


# In[14]:


X_train_l = []
Y_train_l = []
    
# add examples to list by batch
for batch_id, (x_batch, y_batch) in tqdm_notebook(enumerate(train_generator)):
    # Add full batches only - prevent odd array shapes
    if x_batch.shape[0] == BATCH_SIZE:
        X_train_l.append(x_batch)
        Y_train_l.append(y_batch)
    # Break infinite loop manually when required number of batches is reached
    if len(X_train_l) == NB_BATCHES: break

# Sanity check all arrays are same shape
assert len(set(arr.shape for arr in X_train_l)) == 1
assert len(set(arr.shape for arr in Y_train_l)) == 1

# Stack list of arrays
X_train_augm = np.vstack(X_train_l)
Y_train_augm = np.vstack(Y_train_l)

# Sanity check stacking over first dimension
assert X_train_augm.shape[0] == BATCH_SIZE * NB_BATCHES
assert Y_train_augm.shape[0] == BATCH_SIZE * NB_BATCHES


# #### Visualize the augmented data

# In[15]:


'''
plot_imgs_masks(np.squeeze(X_train_augm[:40]), np.squeeze(Y_train_augm[:40]))
'''


# ## Time for the actual model!
# It's a mixture of a U-Net and a DenseNet.

# ### First, an Intersection over Union metric to calculate the accuracy of our identification.
# Remember, IoU helps in object detection by figuring out the similarity (intersection/union) of two bounding boxes.

# In[16]:


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# ### And now the model!

# My findings during testing:
# - Optimizer? **Adam**. Adadelta does best on train and validation but Adam does best on test.
# - Dropout?
# - Image size? **128 pixels** seems to be doing well. (Further testing might help)
# - Number of convblocks? **11 convblocks** is pretty good. (Further testing might help)
# - Reduce learning rate? 
# - Batch size? **32 batch size** works the best.
# - Epochs? The accuracy continues to improve up to **120 epochs**; I could probably test for longer than 120 epochs and still have it improve. (Further testing might help).

# My original U-Net model before melding with DenseNet:
# ```python
# c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
# c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
# p1 = MaxPooling2D((2, 2)) (c1)
# 
# c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
# c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
# p2 = MaxPooling2D((2, 2)) (c2)
# 
# c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
# c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
# p3 = MaxPooling2D((2, 2)) (c3)
# 
# c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
# c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
# p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
# 
# c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
# c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
# 
# u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
# u6 = concatenate([u6, c4])
# c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
# c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
# 
# u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
# u7 = concatenate([u7, c3])
# c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
# c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
# 
# u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
# u8 = concatenate([u8, c2])
# c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
# c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
# 
# u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
# u9 = concatenate([u9, c1], axis=3)
# c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
# c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
# 
# outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
# ```

# Other people's code:
# ```python
# def conv_block(m, dim, acti, bn, res, do=0):
# 	n = Conv2D(dim, 3, activation=acti, padding='same')(m)
# 	n = BatchNormalization()(n) if bn else n
# 	n = Dropout(do)(n) if do else n
# 	n = Conv2D(dim, 3, activation=acti, padding='same')(n)
# 	n = BatchNormalization()(n) if bn else n
# 	return Concatenate()([m, n]) if res else n
# 
# def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
# 	if depth > 0:
# 		n = conv_block(m, dim, acti, bn, res)
# 		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
# 		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
# 		if up:
# 			m = UpSampling2D()(m)
# 			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
# 		else:
# 			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
# 		n = Concatenate()([n, m])
# 		m = conv_block(n, dim, acti, bn, res)
# 	else:
# 		m = conv_block(m, dim, acti, bn, res, do)
# 	return m
# 
# def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
# 		 dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
# 	i = Input(shape=img_shape)
# 	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
# 	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
# 	return Model(inputs=i, outputs=o)
# ```

# Original blocks:
# ```python
# def dense_block(x, bn_axis, channel):
#     x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
#     x1 = Activation('relu')(x1)
#     x1 = Conv2D(channel, (3, 3), padding='same', use_bias=False)(x1)
#     x1 = Conv2D(channel, (3, 3), padding='same', use_bias=False)(x1)
#     x = Concatenate(axis=bn_axis)([x, x1])
#     x = Conv2D(channel, (3, 3), padding='same', use_bias=False)(x)
#     return x
# 
# def up_trans_block(x, bn_axis, channel, dropout=None):
#     t = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
#     t = Activation('relu')(t)
#     c = Conv2D(channel, (3, 3), activation='relu', padding='same') (t)
#     p = AveragePooling2D(2, strides=2)(c)
#     if dropout:
#         d = Dropout(dropout)(p)
#         return d
#     return p
# 
# def down_trans_block(x, y, bn_axis, channel, dropout=None):
#     t = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
#     t = Activation('relu')(t)
#     c = Conv2DTranspose(channel, (2, 2), strides=(2, 2), padding='same') (t)
#     u = Concatenate(axis=bn_axis)([c, y])
#     if dropout:
#         d = Dropout(dropout)(u)
#         return d
#     return u
# ```

# In[ ]:


def dense_block(x, bn_axis, channel, dropout=0):
    x1 = Conv2D(channel, (3, 3), activation='relu', padding='same', use_bias=False)(x)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x1)
    x1 = Dropout(dropout)(x1)   if dropout else x1
    x1 = Conv2D(channel, (3, 3), activation='relu', padding='same', use_bias=False)(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x1)
    x = Concatenate(axis=bn_axis)([x, x1])
    x = Conv2D(channel, (3, 3), padding='same', use_bias=False)(x)
    return x


# In[18]:


def up_trans_block(x, bn_axis, channel, pool=args.pooltype):
    t = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    c = Conv2D(channel, (3, 3), activation='relu', padding='same') (t)
    if pool == 'avg':
        p = AveragePooling2D(2, strides=2)(c)
    else:
        p = MaxPooling2D((2, 2))(c)
    return p


# In[19]:


def down_trans_block(x, y, bn_axis, channel, upsampling=args.upsample):
    t = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    if upsampling:
        c = UpSampling2D()(t)
    else:
        c = Conv2DTranspose(channel, (2, 2), strides=(2, 2), padding='same') (t)
    c = Conv2D(channel, 2, activation='relu', padding='same')(c)
    u = Concatenate(axis=bn_axis)([c, y])
    return u


# Draft 1 of the net:
# ```python
# b1d0 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
# b1d = Concatenate(axis=bn_axis)([b1d0, inputs])
# b1t = up_trans_block(b1d, bn_axis, 16)
# 
# b2d = dense_block(b1t, bn_axis, 16)
# b2t = up_trans_block(b2d, bn_axis, 32)
# 
# b3d = dense_block(b2t, bn_axis, 32)
# b3t = up_trans_block(b3d, bn_axis, 64)
# 
# b4d = dense_block(b3t, bn_axis, 64)
# b4t = down_trans_block(b4d, b3d, bn_axis, 32)
# 
# b5d = dense_block(b4t, bn_axis, 32)
# b5t = down_trans_block(b5d, b2d, bn_axis, 16)
# 
# b6d = dense_block(b5t, bn_axis, 16)
# b6t = down_trans_block(b6d, b1d, bn_axis, 8)
# ```

# Draft 2 of the net:
# ```python
# b1d = dense_block(inputs, bn_axis, 8)
# b1t = up_trans_block(b1d, bn_axis, 8)
# 
# b2d = dense_block(b1t, bn_axis, 16)
# b2t = up_trans_block(b2d, bn_axis, 16)
# 
# b3d = dense_block(b2t, bn_axis, 32)
# b3t = up_trans_block(b3d, bn_axis, 32)
# 
# b4d = dense_block(b3t, bn_axis, 64)
# b4t = up_trans_block(b4d, bn_axis, 64)
# 
# b5d = dense_block(b4t, bn_axis, 128)
# 
# b6t = down_trans_block(b5d, b4d, bn_axis, 64)
# b6d = dense_block(b6t, bn_axis, 64)
# 
# b7t = down_trans_block(b6d, b3d, bn_axis, 32)
# b7d = dense_block(b7t, bn_axis, 32)
# 
# b8t = down_trans_block(b7d, b2d, bn_axis, 16)
# b8d = dense_block(b8t, bn_axis, 16)
# 
# b9t = down_trans_block(b8d, b1d, bn_axis, 8)
# b9d = dense_block(b9t, bn_axis, 8)
# ```

# In[20]:


inputs = Input((IMAGE_SIZE, IMAGE_SIZE, 2))

bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

b1d = dense_block(inputs, bn_axis, 8)
b1t = up_trans_block(b1d, bn_axis, 8)

b2d = dense_block(b1t, bn_axis, 16)
b2t = up_trans_block(b2d, bn_axis, 16)

b3d = dense_block(b2t, bn_axis, 32)
b3t = up_trans_block(b3d, bn_axis, 32)

b4d = dense_block(b3t, bn_axis, 64)
b4t = up_trans_block(b4d, bn_axis, 64)

b5d = dense_block(b4t, bn_axis, 128)
b5t = up_trans_block(b5d, bn_axis, 128)

b6d = dense_block(b5t, bn_axis, 256, DO)

b7t = down_trans_block(b6d, b5d, bn_axis, 128)
b7d = dense_block(b7t, bn_axis, 128)

b8t = down_trans_block(b7d, b4d, bn_axis, 64)
b8d = dense_block(b8t, bn_axis, 64)

b9t = down_trans_block(b8d, b3d, bn_axis, 32)
b9d = dense_block(b9t, bn_axis, 32)

b10t = down_trans_block(b9d, b2d, bn_axis, 16)
b10d = dense_block(b10t, bn_axis, 16)

b11t = down_trans_block(b10d, b1d, bn_axis, 8)
b11d = dense_block(b11t, bn_axis, 8)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (b11d)


# In[21]:


# Create the model
saltModel = Model(inputs=[inputs], outputs=[outputs])


# #### Possible Optimizers:
# ```python
# LR = 3e-3
# Adadelta(lr=LR, rho=0.95, epsilon=1e-6, decay=LR/5)  
# Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay=LR/5)
# SGD(lr = 0.1, decay=1e-4, momentum=0.9, nesterov=True)
# SGD(lr = LR, decay=LR/5, momentum=0.9, nesterov=True)
# RMSprop(lr=LR, rho=0.95, epsilon=1e-6, decay=LR/5)
# ```

# In[22]:


# Compile the model (I'm using Adam optimizer and mean_iou accuracy for now)
saltModel.compile(optimizer=opt, loss='binary_crossentropy', metrics=[mean_iou])


# In[23]:


# Let's get a summary of our model just to know what it's doing
saltModel.summary()


# In[24]:


# And we finally fit the model. Notice that we add an early stopper and a check pointer.
earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', monitor='val_loss', verbose=1, save_best_only=True)
reducelrer = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000005, verbose=1)
cb = []
if args.reducelr == 0:
    cb = [checkpointer]
    print("no reducelr")
else:
    cb = [reducelrer, checkpointer]
    print("reducing lr")

DPT_SIZE = 4
print(X_train_augm.shape)
print(DO)
results = saltModel.fit(X_train_augm,
                    Y_train_augm, 
                    validation_data=(x_valid, y_valid),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=cb)


# ### Visualize the results.

# In[ ]:


'''
fig, (ax_loss, ax_iou) = plt.subplots(1, 2, figsize=(15,5))

_ = ax_loss.plot(results.epoch, results.history["loss"], label="Train loss")
_ = ax_loss.plot(results.epoch, results.history["val_loss"], label="Validation loss")
_ = ax_loss.legend()
_ = ax_loss.set_title('Loss')
#_ = ax_acc.plot(results.epoch, results.history["acc"], label="Train accuracy")
#_ = ax_acc.plot(results.epoch, results.history["val_acc"], label="Validation accuracy")
#_ = ax_acc.legend()
#_ = ax_acc.set_title('Accuracy')
_ = ax_iou.plot(results.epoch, results.history["mean_iou"], label="Train IoU")
_ = ax_iou.plot(results.epoch, results.history["val_mean_iou"], label="Validation IoU")
_ = ax_iou.legend()
_ = ax_iou.set_title('IoU')
'''


# ### Check performance on validation set.

# In[ ]:


saltModel = load_model('model-tgs-salt-1.h5', custom_objects={'mean_iou': mean_iou})
saltModel.evaluate(x_valid, y_valid, verbose=1)


# In[ ]:


preds_valid = saltModel.predict(x_valid, 
                             verbose=1).reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
preds_valid = np.array([downsample(x) for x in preds_valid])
y_valid_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])


# ### Score the model so you can use the best IoU threshold.

# In[ ]:


def iou_metric(labels, y_pred, print_table=False):
    """
    src: https://www.kaggle.com/aglotero/another-iou-metric"""
    class_bins = 2

    # H : ndarray, shape(nx, ny)
    # The bi-dimensional histogram of samples x and y. 
    # Values in x are histogrammed along the first dimension and values in y are histogrammed along the second dimension.
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(class_bins, class_bins))[0] # was 0

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=class_bins)[0] # was 0 (0: no mask, 1: mask)
    area_pred = np.histogram(y_pred, bins=class_bins)[0] # was 0
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


# In[ ]:


def iou_metric_batch(y_true_in, y_pred_in):
    """Compute IoU batchwise"""
    batch_size = y_true_in.shape[0]
    return np.mean([iou_metric(y_true_in[b], y_pred_in[b]) for b in range(batch_size)])


# In[ ]:


thresholds = np.linspace(0.1, 0.9, 40)
ious = np.array([iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) 
                 for threshold in tqdm_notebook(thresholds)])


# In[ ]:


threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]


# In[ ]:


'''
_ = plt.plot(thresholds, ious)
_ = plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
_ = plt.xlabel("Threshold")
_ = plt.ylabel("IoU")
_ = plt.title("Threshold: {} delivers best mean-IoU: {} ".format(threshold_best.round(2), iou_best.round(2)))
_ = plt.legend()
'''


# ## And now it is time to test.

# ### We read in the test set first.

# In[ ]:


x_test = [upsample(np.array(load_img(os.path.join(test_dir, '%s.png' % idx), grayscale=True))) / 255 
                   for idx in tqdm_notebook(test_df.index)]
x_test = np.array(x_test).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
print(x_test.shape)
#np.ones_like(x_test)


# ```python
# [((np.ones_like(x_test[i]) * test_df.loc[i]["z"]) - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
#                      for i in tqdm_notebook(test_df.index)] 
# ```

# In[ ]:


# Create depth layer
x_test_d = [np.ones((128,128,1)) * ((test_df.loc[i]["z"] - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH))
                     for i in tqdm_notebook(test_df.index)]
print(x_test_d[0].shape)
x_test_d = np.array(x_test_d).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
print(x_test_d.shape)


# In[ ]:


'''
train_df["images"] = [np.array(load_img(os.path.join(train_dir_img, '%s.png' % idx), grayscale=True)) / 255 
                      for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img(os.path.join(train_dir_mask, '%s.png' % idx), grayscale=True)) / 255 
                     for idx in tqdm_notebook(train_df.index)]
rain_df["depth"] = [((np.ones_like(train_df.loc[i]["images"]) * train_df.loc[i]["z"]) - MIN_DEPTH)/(MAX_DEPTH - MIN_DEPTH)
                     for i in tqdm_notebook(train_df.index)]
print(train_df["depth"][0].shape)
train_df["images_d"] = [np.dstack((train_df["images"][i], train_df["depth"][i])) for i in tqdm_notebook(train_df.index)]

x_test_d = [np.ones((128,128,1)) * ((test_df.loc[i]["z"] - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH))
                     for i in tqdm_notebook(test_df.index)]
                     
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, depth_train, depth_valid = train_test_split(
    train_df.index.values,
    np.array(train_df.images_d.map(upsample).tolist()).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 2), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1), 
    train_df.z.values,
    test_size=0.2, 
    stratify=train_df.masks.map(coverage_class), 
    random_state=1)
'''


# In[ ]:


# Predict on train, val and test
saltModel = load_model('model-tgs-salt-1.h5', custom_objects={'mean_iou': mean_iou})


# In[ ]:


print(x_test.shape)
print(x_test[0].shape)
print(x_test_d.shape)
print(x_test_d[0].shape)
x_test_full = [np.dstack((elem, x_test_d[idx])) for idx, elem in enumerate(x_test)]
print(x_test_full[0].shape) 
x_test_full = np.array(x_test_full)
print(x_test_full.shape)


# In[ ]:


preds_test = saltModel.predict(x_test_full)


# In[ ]:


test_ids = next(os.walk(test_dir))[2]
assert len(set(test_ids) ^ set(test_df.index+'.png')) == 0


# Original rlenc
# ```python
# def RLenc(img, order='F', format=True):
#     """
#     img is binary mask image, shape (r,c)
#     order is down-then-right, i.e. Fortran
#     format determines if the order needs to be preformatted (according to submission rules) or not
# 
#     returns run length as an array or string (if format is True)
#     """
#     bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
#     runs = []  ## list of run lengths
#     r = 0  ## the current run length
#     pos = 1  ## count starts from 1 per WK
#     for c in bytes:
#         if (c == 0):
#             if r != 0:
#                 runs.append((pos, r))
#                 pos += r
#                 r = 0
#             pos += 1
#         else:
#             r += 1
# 
#     # if last run is unsaved (i.e. data ends with 1)
#     if r != 0:
#         runs.append((pos, r))
#         pos += r
#         r = 0
# 
#     if format:
#         z = ''
# 
#         for rr in runs:
#             z += '{} {} '.format(rr[0], rr[1])
#         return z[:-1]
#     else:
#         return runs
# ```

# In[ ]:


def RLenc(img, order='F'):
    """Convert binary mask image to run-length array or string.
    
    Args:
    img: image in shape [n, m]
    order: is down-then-right, i.e. Fortran(F)
    string: return in string or array

    Return:
    run-length as a string: <start[1s] length[1s] ... ...>
    """
    bytez = img.reshape(img.shape[0] * img.shape[1], order=order)
    bytez = np.concatenate([[0], bytez, [0]])
    runs = np.where(bytez[1:] != bytez[:-1])[0] + 1 # pos start at 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Use for sanity check the encode function
def RLdec(rl_string, shape=(101, 101), order='F'):
    """Convert run-length string to binary mask image.
    
    Args:
    rl_string: 
    shape: target shape of array
    order: decode order is down-then-right, i.e. Fortran(F)

    Return:
    binary mask image as array
    """
    s = rl_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order=order)


# In[ ]:


pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) > threshold_best)) 
             for i, idx in enumerate(tqdm_notebook(test_df.index.values))}


# ## Lastly, we prepare the submission.

# The submission naming process before splitting:
# ```python
# 'saltSubmission_8.14_%s_batch%s_epoch%s.csv' % (optimizer, BATCH_SIZE, EPOCHS)
# ```

# In[ ]:


sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']

sub.to_csv(os.path.join(model_dir, 'saltSubmission_' + prefix + info))

print('Prediction result saved as saltSubmission_' + prefix + info)

sub.head()

