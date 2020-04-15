import numpy
import config
import os
from glob import glob
import numpy as np
import keras
import random
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import glob
import math
import cv2
from exceptions import CustomError
import pandas
from keras.models import Model as mp
from keras.layers import Average, Input
from models import Model

def getImages(directory):

    '''
    THIS FUNTION RETRIEVES ALL IMAGES FILES
    :param directory: str --> dict/*.jpg
    :return: list of all jpg files
    '''

    try:

        return sorted(glob.glob(directory))

    except:
        raise

def addNewColumn_Populate_DataFrame(dataFrame, name_new_column, dataToPopulate):

    '''
    THIS FUNCTION IS USED TO ADD NEW COLUMN TO DATAFRAME, AND POPULATE COLUMN WITH DATA
    :param dataFrame: dataFrame --> dataFrame to apply changes
    :param name_new_column: str --> name of new column
    :param dataToPopulate: List (str) --> strings to populate data
    :return: dataFrame changed
    '''

    try:

        dataFrame[name_new_column] = dataToPopulate
        return dataFrame

    except:
        raise

def getX_Y_Image(image_path : str):

    '''
    THIS FUCNTION IS USED TO RETIEVE X (pixel RGB VALUES) of an image
    :param image_path: str --> image path of image
    :return: X: numpy array --> RGB VALUES OF IMAGE
    '''

    try:

        image = cv2.imread(image_path)
        X = cv2.resize(image, (config.HEIGHT, config.WIDTH), interpolation=cv2.INTER_CUBIC)

        return X

    except:
        raise

def getDataFromImages(dataframe : pandas.DataFrame, size):

    '''
    THIS FUNCTION IS USED TO RETRIEVE X and Y data inherent from all images
    :param dataframe: Pandas Dataframe --> with all images path's and correspondent targets
    :param size: integer --> this values is <= than total_images
            e.g = total images equals to 10000 and user only wants 5000
            note: stratify option is used to continue with respective perecntage of samples by class
    :return: X : numpy array --> Data from images (pixels from images)
    :return Y: numpy array --> targets for each image
    '''

    try:

        ##GET TOTAL IMAGES
        number_images = dataframe.shape[0]
        X = []
        Y = []

        d = dict(enumerate(dataframe.dx.cat.categories))
        numeric_targets = dataframe.dx.cat.codes.values

        if size > number_images:
            raise

        elif size < number_images:
            ## GET PERCENTAGE OF IMAGES BY CLASS
            images_by_class = [int(round(((dataframe.loc[dataframe.dx == config.DICT_TARGETS[i], config.DX].count()) / number_images)*size))
                                   for i in range(len(dataframe.dx.unique()))]

            counter_by_class = [config.DICT_TARGETS[i] for i in range(len(dataframe.dx.unique()))]
            for i in range(dataframe.shape[0]):
                target = dataframe.at[i, config.DX] # GET TARGET OF THIS IMAGE
                index_target_counter = counter_by_class.index(target) # CORRESPONDENT INDEX BETWEEN CLASS AND NUMBER OF PERMITTED IMAGES FOR THIS CLASS
                if images_by_class[index_target_counter] != 0: ## IF THIS CLASS STILL ALLOWS TO PUT IMAGES
                    X.append(getX_Y_Image(dataframe.at[i, config.PATH]))
                    Y.append(numeric_targets[i])
                    images_by_class[index_target_counter] = images_by_class[index_target_counter] - 1 # DECREASE NUMBER OF IMAGES ALLOWED FOR THIS CLASS
                else:
                    continue
                if all(images_by_class[i] == 0 for i in range(len(images_by_class))): ## IF JOB IS FINISHED --> ALREADY HAVE STRATIFIED IMAGES FOR ALL CLASSES
                    break

            return np.array(X), np.array(Y)

        else: ## size == number_images, i want all images

            for i in range(dataframe.shape[0]):
                X.append(getX_Y_Image(dataframe.at[i, config.PATH]))
                Y.append(numeric_targets[i])

            return np.array(X), np.array(Y)
    except:
        raise CustomError.ErrorCreationModel(config.ERROR_ON_GET_DATA)

def impute_null_values(dataFrame, column, mean=True):

    '''
    THIS FUNCTION IS USED TO RETRIEVE NULL VALUES ON DATAFRAME
    :param dataFrame: dataFrame
    :param column: str --> name of column to impute null values
    :param mean: boolean (Default = True) --> if True impute with mean of column, if False apply median to null values
    :return: DataFrame --> changed DataFrame
    '''

    try:

        series_column = dataFrame[column] ## GET SERIES COLUMN

        if len(series_column) == 0: ## INVALID COLUMN NAME --> len is more quickly than empty
            return dataFrame

        if mean == True:
            column_mean = series_column.mean()  ## GET MEAN OF COLUMN
            mean = math.trunc(column_mean)
            dataFrame = dataFrame.fillna(mean)
            return dataFrame

        column_median = series_column.median()
        median = math.trunc(column_median)
        dataFrame = dataFrame.fillna(median)
        return dataFrame

    except:
        raise

def resize_images(width, height, data):

    '''
    :param width: int --> pixel width to resize image
    :param height: int --> pixel height to resize image
    :param data: dataframe --> shape ["id", "image_path", "target"]
    :return x: numpy array --> shape (number images, width, height)
    :return y: numpy array --> shape (number images, target)
    '''

    try:

        x = []
        y = []

        for i in range(len(data[config.ID])):
            image = cv2.imread(data.at[i, config.IMAGE_PATH])
            x.append(cv2.resize(image, (width, height)))
            y.append(data.at[i, config.TARGET])

        return numpy.array(x), numpy.array(y)

    except:
        raise

def normalize(X_train, X_val, X_test):

    '''
    #REF https://forums.fast.ai/t/images-normalization/4058/8
    :param X_train: numpy array representing training data
    :param X_val: numpy array representing validation data
    :param X_test: numpy array representing test data
    :return X_train: numpy array normalized
    :return X_val: numpy array normalized
    :return X_X_test: numpy array normalized
    '''

    try:

        mean = np.mean(X_train, axis=config.STANDARDIZE_AXIS_CHANNELS) #STANDARDIZE BY CHANNELS
        std = np.std(X_train, axis=config.STANDARDIZE_AXIS_CHANNELS) #STANDARDIZE BY CHANNELS
        X_train = (X_train-mean)/(std+1e-7)
        X_val = (X_val-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)

        # minmax_scale = preprocessing.MinMaxScaler().fit(X_train)
        # X_train = minmax_scale.transform(X_train)
        # X_val = minmax_scale.transform(X_val)
        # X_test = minmax_scale.transform(X_test)
        #
        # #RESHAPE AGAIN TO 4D
        # shape_data = (X_train.shape[0], config.WIDTH, config.HEIGHT, config.CHANNELS)
        # X_train = X_train.reshape(shape_data)
        # shape_data = (X_val.shape[0], config.WIDTH, config.HEIGHT, config.CHANNELS)
        # X_val = X_val.reshape(shape_data)
        # shape_data = (X_test.shape[0], config.WIDTH, config.HEIGHT, config.CHANNELS)
        # X_test = X_test.reshape(shape_data)

        return X_train, X_val, X_test
    except:
        raise

def one_hot_encoding(y_train, y_val, y_test):

    '''

    :param y_train: numpy array with training targets
    :param y_val: numpy array with validation targets
    :param y_test: numpy array with test targets
    :return y_train: numpy array categorized [1 0] --> class 0 or [0 1] --> class 1
    :return y_val: numpy array categorized
    :return y_test: numpy array categorized
    '''

    try:

        y_train = keras.utils.to_categorical(y_train, config.NUMBER_CLASSES)
        y_val = keras.utils.to_categorical(y_val, config.NUMBER_CLASSES)
        y_test = keras.utils.to_categorical(y_test, config.NUMBER_CLASSES)

        return y_train, y_val, y_test

    except:
        raise

def decode_array(array):

    '''
    THIS FUNCTION IS USED TO DECODE ENCODING ARRAY'S LIKE PREDICTIONS RESULTED FROM MODEL PREDICT
    e.g : array[[0 1]
                [1 0]]
        return array[[1]
                     [0]]
    :param array: numpy array
    :return: numpy array --> decoded array
    '''

    try:

        decoded_array = np.argmax(array, axis=1) #RETURNS A LIST

        return decoded_array
    except:
        raise

def getConfusionMatrix(predictions, y_test, dict):

    '''
    THIS FUNCTION IS USED IN ORDER TO SHOW MAIN RESULTS OF MODEL EVALUATION (ACCURACY, RECALL, PRECISION OR F-SCORE)
    :param predictions: numpy array --> model predictions
    :param y_test: numpy array --> real targets of test data
    :return: report: dict --> with metrics results (ACCURACY, RECALL, PRECISION OR F-SCORE)
    :return: confusion_mat: ndarray (n_classes, n_classes)
    '''

    try:

        #CREATE REPORT
        if dict == True:
            report = classification_report(y_test, predictions, target_names=config.DICT_TARGETS,
                                       output_dict=True) # returns a dict with metrics to access easily, important in optimizer
        else:
            report = classification_report(y_test, predictions, target_names=config.DICT_TARGETS)
        #CREATION OF CONFUSION MATRIX
        confusion_mat = confusion_matrix(y_test, predictions)

        return report, confusion_mat
    except:
        raise

def plot_cost_history(history):

    '''
    THIS FUNNCTION PLOTS COST HISTORY
    :param history: history object resulted from train
    :return: none --> only plt show
    '''

    try:

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    except:
        raise

def plot_accuracy_plot(history):

    '''
    THIS FUNNCTION PLOTS ACCURACY HISTORY
    :param history: history object resulted from train
    :return:
    '''

    try:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    except:
        raise

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def lr_scheduler(epoch):
    return config.LEARNING_RATE * (0.5 ** (epoch // config.DECAY))

def defineMask(maskValues):

    '''
    THIS FUNCTION IS USED TO CREATE BLACK/WHITE MASKS TO AN IMAGE
    :param maskValues: numpy array --> predicted image mask values (pixel)
    :return: black/white mask corresponding to the image predicted defined before (on model.predict --> to specific image)
    '''

    try:


        for i in range(maskValues.shape[0]):
            for j in range(maskValues.shape[1]):
                if maskValues[i][j] > 0.5:
                    maskValues[i][j] = 1
                else:
                    maskValues[i][j] = 0

        return maskValues

    except:
        raise

def concate_image_mask(image, mask):

    try:

        ## reshape to 2d
        mask = np.array(mask.reshape(config.WIDTH, config.HEIGHT), dtype=np.uint8)

        ## bitwise image and mask
        res = cv2.bitwise_and(image, image, mask=mask)

        return res

    except:
        raise

def ensemble(models):

    '''
    THIS FUNCTION IS USED TO ENSEMBLE OUTPUT OF A LIST OF MODELS, CONSIDERING ITS AVERAGE
    :param models: List Models : models used (AlexNet, VGGNet, ResNet)
    :return: model: model with average outputs of all models considered
    '''

    try:

        ## get outputs of each model
        input_shape = (config.WIDTH, config.HEIGHT, config.CHANNELS)
        input_model = Input(input_shape)
        models_out = [i(input_model) for i in models]

        ## get average of each model (ensemble)
        average = Average() (models_out)

        ## define model with new outputs

        model = mp(input_model, average, name='ensemble')

        return model

    except:
        raise

def print_final_results(y_test, predictions, history, dict=False):

    '''
    THIS FUNCTION IS USED TO PRINT ANND PLOT FINAL RESULTS OF MODEL EVALUATION
    :param y_test: real predictions of test
    :param predictions: numpy array : predictions of model
    :param history: History.history : history of train (validation and train along epochs)
    :return: nothing only print's and plot's
    '''

    try:

        if history != None:
            print(plot_cost_history(history))
            print(plot_accuracy_plot(history))
        predictions = decode_array(predictions)  # DECODE ONE-HOT ENCODING PREDICTIONS ARRAY
        y_test_decoded = decode_array(y_test)  # DECODE ONE-HOT ENCODING y_test ARRAY
        report, confusion_mat = getConfusionMatrix(predictions, y_test_decoded, dict)
        print(report)
        plt.figure()
        plot_confusion_matrix(confusion_mat, config.DICT_TARGETS)

    except:
        raise