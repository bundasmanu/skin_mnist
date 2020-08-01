import numpy as np
import itertools

# global counter
counter_iterations = itertools.count(start=0, step=1)

# image dimensions
WIDTH = 128
HEIGHT = 128
CHANNELS = 3

STANDARDIZE_AXIS_CHANNELS = (0,1,2)
NUMBER_CLASSES = 7

## FILES
INPUT_FOLDER = 'input'
IMAGES_ACESS = 'images/*.jpg'
INFO_PATIENTS = 'HAM10000_metadata.csv'
OTHER_CSV = 'HMNIST_8_8_l.csv'
UNET_WIGHTS_PATH = INPUT_FOLDER+'/unet-weights/unet_100_epoch.h5'
MASKS_FOLDER = 'masks'
PREDICTIONS_MASKS_TRAIN = 'predictions_masks_train.bin'
TEMP_ARRAYS = 'temp_arrays/' + PREDICTIONS_MASKS_TRAIN

#DATAFRAME COLUMNS
LESION_ID = 'lesion_id'
IMAGE_ID = 'image_id'
DX = 'dx'
DX_TYPE = 'dx_type'
AGE = 'age'
SEX = 'sex'
LOCALIZATION = 'localization'
PATH = 'path'

DICT_TARGETS = (
    'akiec' ,
    'bcc' ,
    'bkl' ,
    'df' ,
    'mel' ,
    'nv',
    'vasc'
)

WANTED_IMAGES = 10015

LEARNING_RATE = 0.001
DECAY = 1e-6

RELU_FUNCTION = "relu"
SOFTMAX_FUNCTION = "softmax"
SIGMOID_FUNCTION = "sigmoid"

VALIDATION_SPLIT = 0.15 # 15%
TEST_SPLIT = 0.2353 # 20%
RANDOM_STATE = 0

LOSS_BINARY = "binary_crossentropy"
LOSS_CATEGORICAL = "categorical_crossentropy"

VALID_PADDING = "valid"
SAME_PADDING = "same"

ACCURACY_METRIC = "accuracy"
VALIDATION_ACCURACY = "val_accuracy"

BATCH_SIZE_ALEX_NO_AUG = 32
BATCH_SIZE_ALEX_AUG = 32
EPOCHS = 16
MULTIPROCESSING = True
SHUFFLE = True

GLOROT_SEED = 0
HE_SEED = 0

X_VAL_ARGS = "X_Val"
Y_VAL_ARGS = "y_val"

HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ROTATION_RANGE = 10
ZOOM_RANGE = 0.25
BRITNESS_RANGE= 0.3

#FLAGS TRAIN STRATEGY
UNDERSAMPLING = True
OVERSAMPLING = False
DATA_AUGMENTATION = False

PSO_OPTIMIZER = "PSO"
GA_OPTIMIZER = "GA"

ALEX_NET = "ALEXNET"
VGG_NET = "VGGNET"
RES_NET = "RESNET"
DENSE_NET = "DENSENET"
U_NET = "UNET"

FLAG_SEGMENT_IMAGES = 0

#EXCEPTIONS MESSAGES
ERROR_MODEL_EXECUTION = "\nError on model execution"
ERROR_NO_ARGS = "\nPlease provide args: ",X_VAL_ARGS," and ", Y_VAL_ARGS
ERROR_NO_ARGS_ACCEPTED = "\nThis Strategy doesn't accept more arguments"
ERROR_NO_MODEL = "\nPlease pass a initialized model"
ERROR_INVALID_OPTIMIZER = "\nPlease define a valid optimizer: ", PSO_OPTIMIZER," or ", GA_OPTIMIZER
ERROR_INCOHERENT_STRATEGY = "\nYou cannot choose the oversampling and undersampling strategies at the same time"
ERROR_ON_UNDERSAMPLING = "\nError on undersampling definition"
ERROR_ON_OVERSAMPLING = "\nError on oversampling definition"
ERROR_ON_DATA_AUG = "\nError on data augmentation definition"
ERROR_ON_TRAINING = "\nError on training"
ERROR_ON_OPTIMIZATION = "\nError on optimization"
ERROR_INVALID_NUMBER_ARGS = "\nPlease provide correct number of args"
ERROR_ON_BUILD = "\nError on building model"
ERROR_APPEND_STRATEGY = "\nError on appending strategy"
ERROR_ON_PLOTTING = "\nError on plotting"
ERROR_ON_GET_DATA = "\nError on retain X and Y data"
ERROR_ON_IDENTITY_BLOCK ="\nError on modelling identity block, please check the problem"
ERROR_ON_CONV_BLOCK ="\nError on modelling convolutional block, please check the problem"
ERROR_ON_UNET_STRATEGY = "\nError on U-Net strategy applying"

#PSO OPTIONS
TOPOLOGY_FLAG = 0 # 0 - GBest , 1 - LBest
PARTICLES = 20
ITERATIONS = 10
gbestOptions = {'w' : 0.7, 'c1' : 1.4, 'c2' : 1.4}
lbestOptions = {'w' : 0.7, 'c1' : 1.4, 'c2' : 1.4, 'k' : 2, 'p' : 2}

#GA OPTIONS
TOURNAMENT_SIZE = 100
INDPB = 0.6
CXPB = 0.4
MUTPB = 0.2

## weight load files
UNET_WEIGHTS_100 = 'unet_100_epoch.h5'
UNET_BUNET_WEIGHTS = 'weight_isic18.hdf5'
VGG_NET_WEIGHTS_FILE = 'vggnet_weights.h5'
ALEX_NET_WEIGHTS_FILE = 'alexnet_weights.h5'
RES_NET_WEIGHTS_FILE = 'resnet_weights.h5'

## u-net backbone
BACKBONE = 'resnet34'

## class weights
class_weights={
    0: 2.0, # akiec
    1: 1.5, # bcc
    2: 1.5, # bkl
    3: 3.0, # df
    4: 3.0, # mel # Try to make the model more sensitive to Melanoma.
    5: 1.0, # nv
    6: 1.0, # vasc
}

class_sampling={ # oversampling
    0: 2179, # akiec
    1: 2179, # bcc
    2: 2179, # bkl
    3: 2179, # df
    4: 2179, # mel # Try to make the model more sensitive to Melanoma.
    5: 4358, # nv
    6: 2179, # vasc
}

class_sampling2={
    0: 213,  # akiec
    1: 334,  # bcc
    2: 714,  # bkl
    3: 75,  # df
    4: 723,  # mel # Try to make the model more sensitive to Melanoma.
    5: 2179,  # nv
    6: 92,  # vasc
}

# PSO BOUNDS LIMITS
MAX_VALUES_LAYERS_ALEX_NET = [3.99, 3.99, 96, 48, 2.99, 96, 64] # nº of normal conv's, nº of stack cnn layers, nº of feature maps of initial conv, growth rate, nº neurons of FCL layer and batch size
MIN_VALUES_LAYERS_ALEX_NET = [1, 1, 4, 0, 1, 14, 12]
MAX_VALUES_LAYERS_VGG_NET = [7.99, 96, 48, 2.99, 72, 64] # nº of stack cnn layers, nº of feature maps of initial conv, growth rate, nº neurons of FCL layer and batch size
MIN_VALUES_LAYERS_VGG_NET = [2, 4, 0, 1, 14, 12]
MAX_VALUES_LAYERS_RESNET_NET = [96, 5.99, 2.99, 48, 64] # number of filters of first conv layer, number of conv+identity blocks, growth rate and batch size
MIN_VALUES_LAYERS_RES_NET = [4, 1, 0, 0, 12]
MAX_VALUES_LAYERS_DENSE_NET = [96, 5.99, 6.99, 32, 1, 64] # nº of initial filters, nº of dense blocks, nº of composite blocks, growth rate, compression rate and batch size
MIN_VALUES_LAYERS_DENSE_NET = [4, 1, 2, 2, 0.1, 12]

#FILENAME POSITION PSO VARIATION
POS_VAR_LOWER = 'particlesPso.mp4'
POS_VAR_INTER = 'particlesPso_intermedia.mp4'
POS_VAR_HIGHTER = 'particlesPso_elevada.mp4'
POS_VAR_EXP = 'pos_var_exp.html'

# VARIABLES MAKE .mp4 VIDEO with particles movement position
X_LIMITS = [1, 256]
Y_LIMITS = [1, 256]
LABEL_X_AXIS = 'Nºfiltros 1ªcamada'
LABEL_Y_AXIS = 'Nºfiltros 2ªcamada'

# PSO INIT DEFINITIONS --> IN ARGS FORM
pso_init_args_alex = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    7, # dimensions (6nº of normal conv's, nº of stack cnn layers, nº of feature maps of initial conv, growth rate, nº of FCL layers that preceding Output layer, nº neurons of FCL layer (equals along with each other) and batch size)
    np.array(MIN_VALUES_LAYERS_ALEX_NET),
    np.array(MAX_VALUES_LAYERS_ALEX_NET)  # superior bound limits for dimensions
)

pso_init_args_vgg = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    6,  # dimensions (nº of stack cnn layers, nº of feature maps of initial conv, growth rate, nº of FCL layers that preceding Output layer, nº neurons of FCL layer (equals along with each other) and batch size)
    np.array(MIN_VALUES_LAYERS_VGG_NET),
    np.array(MAX_VALUES_LAYERS_VGG_NET)  # superior bound limits for dimensions
)

pso_init_args_resnet = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    5,  # number of filters of first conv layer, number of conv+identity blocks, nº of identity blocks, growth rate and batch size
    np.array(MIN_VALUES_LAYERS_RES_NET),
    np.array(MAX_VALUES_LAYERS_RESNET_NET)  # superior bound limits for dimensions
)

pso_init_args_densenet = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    6,  # dimensions (init Conv Feature Maps, number of blocks, number cnn layers on blocks, growth rate, comprension rate and batch size)
    np.array(MIN_VALUES_LAYERS_DENSE_NET), # lower bound limits for dimensions
    np.array(MAX_VALUES_LAYERS_DENSE_NET)  # superior bound limits for dimensions
)

## verbose and summary options on build and train
TRAIN_VERBOSE = 1 # 0 - no info, 1- info, 2- partial info
BUILD_SUMMARY = 1 # 0 - no summary, 1- summary
