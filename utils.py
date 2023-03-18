# from collections import defaultdict
import os
import numpy as np
# from keras import backend as K

# import keras
# from keras import preprocessing
from keras.layers import Input
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3

# from keras_preprocessing import image
from keras.preprocessing import image
from keras.applications.vgg16 import decode_predictions
from keras.applications.resnet50 import preprocess_input as ppi224
from keras.applications.inception_v3 import preprocess_input as ppi299

from Components import Dimension

from scipy.misc import imsave
# from keras.preprocessing.image import save_img

def load_model(target_model):
    if target_model == 0:
        model = VGG16(input_tensor=Input(shape=(224, 224, 3)))
        img_rows, img_cols, img_rgb = 224, 224, 3
    elif target_model == 1:
        model = VGG19(input_tensor=Input(shape=(224, 224, 3)))
        img_rows, img_cols, img_rgb = 224, 224, 3
    elif target_model == 2:
        model = ResNet50(input_tensor=Input(shape=(224, 224, 3)))
        img_rows, img_cols, img_rgb = 224, 224, 3
    elif target_model == 3:
        model = InceptionV3(input_tensor=Input(shape=(299, 299, 3)))
        img_rows, img_cols, img_rgb = 299, 299, 3
    print("model load done")
    return model, img_rows, img_cols, img_rgb

# create and initialize the dimension
def initialize_Dim(DimSize, MaxValue, MinValue):
    dim = Dimension()
    dim.setDimensionSize(DimSize)
    dim.setMax(MaxValue)
    dim.setMin(MinValue)
    return dim

# load the seed set
def initialize_img_set(path, img_ext):
    img_set = image.list_pictures(path, ext=img_ext)
    img_set.sort()
    return img_set

def initialize_log_file(filename, args):
    # log file
    f = open("./log/"+filename, 'w')
    # record the parameter
    f.write('tar_model:'+str(args.target_model)+'\n')
    f.write('max_perturbation:'+str(args.max_perturbation)+'\n')
    f.write('samples:'+str(args.samples)+'\n')
    f.write('iterations:'+str(args.iterations)+'\n')
    f.write('samples:'+str(args.samples)+'\n')
    f.write('uncertain_bits:'+str(args.uncertain_bits)+'\n')
    f.write('positive_num:'+str(args.positive_num)+'\n')
    f.write('target:'+str(args.target)+'\n')
    f.write('target_index:'+str(args.target_index)+'\n')
    f.write('time_out:'+str(args.time_out)+'\n')
    f.write('resize:'+str(args.resize)+'\n')
    f.write('resize_mode:'+str(args.resize_mode)+'\n')
    f.write('\n')
    f.close()

# set up the folfer to save imgs
def mkdir_folders(filename):
    root_path = "./Results/"+filename+"/"
    success_path = root_path + 'success/'
    fail_path = root_path + 'fail/'
    npy_path = root_path + 'npy/'
    os.makedirs(root_path)
    os.makedirs(success_path)
    os.makedirs(fail_path)
    os.makedirs(npy_path)
    return root_path, success_path, fail_path, npy_path


def img_to_input(img_path, size=299):
    img = image.load_img(img_path, target_size=(size, size))
    input_img_data = image.img_to_array(img)
    return input_img_data   #shape (img_rows, img_cols, img_rgb), type float

'''
    img_input: shape (batch_size, img_rows, img_cols, img_rgb)
'''
def input_to_prediction(img_input, model, batch_size=1):

    input_img_data = img_input.copy()
    dim_size = input_img_data.shape[1]
    # input_img_data = np.expand_dims(copy_input, axis=0)  #shape (0, 224, 224, 3) type float32
    if dim_size == 224:
        input_img_data = ppi224(input_img_data)
    elif dim_size == 299:
        input_img_data = ppi299(input_img_data)
    pred = model.predict(input_img_data, batch_size=batch_size)
    return pred

def max_abs(list_):
    positive = max(list_)
    negative = min(list_)
    if abs(negative) < abs(positive):
        return abs(positive)
    else:
        return abs(negative)

def decode_label(pred):
    return decode_predictions(pred)[0][0][1]


def find_k_max(pred, k):
    temp = pred.copy()
    temp = np.argsort(temp)
    for i in range(k):
        temp_label = temp[0][-1-i]
        print(temp_label, pred[0][temp_label])
    return temp[0][-1-k]
	 
def noise_resize(perturbation, dimsize, imgrows, imgrgb, eps, resize_mode):
    temp_noise = np.array(perturbation.getFeatures()).reshape(dimsize, dimsize, imgrgb)
    temp_noise = np.around(temp_noise+eps).astype('uint8')
    imsave('./noise.png', temp_noise)
    noise = image.img_to_array(image.load_img('./noise.png', target_size=(imgrows, imgrows), interpolation=resize_mode))
    noise -= eps
    return noise

def save_img(img, npypath, imgpath):
    np.save(npypath, img)
    img = img.astype('uint8')
    imsave(imgpath, img)
