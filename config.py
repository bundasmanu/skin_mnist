WIDTH = 90
HEIGHT = 75
CHANNELS = 3

STANDARDIZE_AXIS_CHANNELS = (0,1,2,3)
NUMBER_CLASSES = 7

## FILES
INPUT_FOLDER = 'input'
IMAGES_ACESS = 'images/*.jpg'
INFO_PATIENTS = 'HAM10000_metadata.csv'
OTHER_CSV = 'HMNIST_8_8_l.csv'

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
    'bkl' ,
    'akiec' ,
    'nv' ,
    'mel' ,
    'bcc' ,
    'vasc',
    'df'
)

WANTED_IMAGES = 10015

VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
RANDOM_STATE = 0

MULTIPROCESSING = True

X_VAL_ARGS = "X_Val"
Y_VAL_ARGS = "y_val"

PSO_OPTIMIZER = "PSO"
GA_OPTIMIZER = "GA"

ALEX_NET = "ALEXNET"
VGG_NET = "VGGNET"

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

#PSO OPTIONS
TOPOLOGY_FLAG = 0
PARTICLES = 1
ITERATIONS = 2
PSO_DIMENSIONS = 5
TOPOLOGY_FLAG = 0 # 0 MEANS GBEST, AND 1 MEANS LBEST
gbestOptions = {'w' : 0.9, 'c1' : 0.3, 'c2' : 0.3}
lbestOptions = {'w' : 0.9, 'c1' : 0.3, 'c2' : 0.3, 'k' : 4, 'p' : 2}

#GA OPTIONS
TOURNAMENT_SIZE = 100
INDPB = 0.6
CXPB = 0.4
MUTPB = 0.2
