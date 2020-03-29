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

def getImages(directory):

    '''
    THIS FUNTION RETRIEVES ALL IMAGES FILES
    :param directory: str --> dict/*.jpg
    :return: list of all jpg files
    '''

    try:

        return glob.glob(directory)

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

        mean = np.mean(X_train,axis=config.STANDARDIZE_AXIS_CHANNELS) #STANDARDIZE BY CHANNELS
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
        y_val =  keras.utils.to_categorical(y_val, config.NUMBER_CLASSES)
        y_test =  keras.utils.to_categorical(y_test, config.NUMBER_CLASSES)

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

def getConfusionMatrix(predictions, y_test):

    '''
    THIS FUNCTION IS USED IN ORDER TO SHOW MAIN RESULTS OF MODEL EVALUATION (ACCURACY, RECALL, PRECISION OR F-SCORE)
    :param predictions: numpy array --> model predictions
    :param y_test: numpy array --> real targets of test data
    :return: report: dict --> with metrics results (ACCURACY, RECALL, PRECISION OR F-SCORE)
    :return: confusion_mat: ndarray (n_classes, n_classes)
    '''

    try:

        #CREATE REPORT
        report = classification_report(y_test, predictions, target_names=config.LIST_CLASSES_NAME)

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