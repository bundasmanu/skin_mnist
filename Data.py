import numpy as np
import config

class Data:

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def decodeYData(self):

        '''
        THIS FUNCTION IS USED TO DISABLE ONE-HOT-ENCODING
        e.g --> y_train = [ [0 1] [1 0] ]
                return [ [1] [0] ]
        :param y_train: numpy array: training targets
        :return: numpy array: decoded training targets
        '''

        try:

            y_train = [np.argmax(self.y_train[i], axis=0) for i in range(self.y_train.shape[0])]
            y_train = np.array(y_train)

            return y_train

        except:
            raise

    def reshape4D_to_2D(self):

        '''
        THIS FUNCTION IS USED TO RESHAPE TRAINING DATA FROM 4D TO 2D --> IS NEED TO APPLY STRATEGIES
        :param X_train: numpy array --> training data 4D (SAMPLES, WIDTH, HEIGHT, CHANNELS)
        :return: numpy array --> training data 2D (SAMPLES, FEATURES) --> FEATURES = (WIDTH * HEIGHT * CHANNELS)
        '''

        try:

            feature_reshape = (self.X_train.shape[1] * self.X_train.shape[2] * self.X_train.shape[3])
            X_train = self.X_train.reshape(self.X_train.shape[0], feature_reshape)

            return X_train

        except:
            raise

    def reshape2D_to_4D(self):

        '''
        THIS FUNCTION IS USED TO RESHAPE TRAINING DATA FROM 2D TO 4D --> IS NEED TO APPLY STRATEGIES
        :param X_train: numpy array --> training data 2D (SAMPLES, FEATURES) --> FEATURES = (WIDTH * HEIGHT * CHANNELS)
        :return: numpy array --> training data 4D  (SAMPLES, WIDTH, HEIGHT, CHANNELS)
        '''

        try:

            shape_data = (self.X_train.shape[0], config.WIDTH, config.HEIGHT, config.CHANNELS)
            X_train = self.X_train.reshape(shape_data)

            return X_train

        except:
            raise