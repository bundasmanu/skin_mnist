from . import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Dense, Flatten, Input
import config
from exceptions import CustomError
from .Strategies_Train import Strategy
from keras.optimizers import Adam
from keras.callbacks.callbacks import History
from typing import Tuple
import Data
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import config_func
from sklearn.utils import class_weight
import numpy
from keras.models import Model as mp

class AlexNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(AlexNet, self).__init__(data, *args)

    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        return super(AlexNet, self).addStrategy(strategy)
    
    def add_conv(self, input, numberFilters, dropoutRate, input_shape=None):

        '''
        This function represents the use of individual convolutional layers (initial layers on AlexNet)
        Conv --> Activation --> MaxPooling --> BatchNormalization --> Dropout
        :param input: tensor with current model architecture
        :param numberFilters: integer: number of filters to put on Conv layer
        :param dropoutRate: float (between 0.0 and 1.0)
        :param input_shape: tuple (height, width, channels) with shape of first cnn layer --> default None (not initial layers)
        :return: tensor of updated model
        '''

        try:

            if input_shape != None:
                input = Conv2D(filters=numberFilters, input_shape=input_shape, kernel_size=(3,3), strides=2,
                               padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            else:
                input = Conv2D(filters=numberFilters, kernel_size=(3, 3), strides=1,
                               padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY))(input)

            input = Activation(config.RELU_FUNCTION) (input)
            input = MaxPooling2D(pool_size=(2,2), strides=2) (input)
            input = BatchNormalization() (input)
            input = Dropout(dropoutRate) (input)

            return input

        except:
            raise

    def add_stack(self, input, numberFilters, dropoutRate):

        '''
        This function represents the implementation of stack cnn layers (in this case using only 2 cnn layers compacted)
        Conv --> Activation --> Conv --> Activation --> MaxPooling --> BatchNormalization --> Dropout
        :param input: tensor with current model architecture
        :param numberFilters: integer: number of filters to put on Conv layer
        :param dropoutRate: float (between 0.0 and 1.0)
        :return: tensor of updated model
        '''

        try:

            input = Conv2D(filters=numberFilters, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING,
                           kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            input = Activation(config.RELU_FUNCTION) (input)
            input = Conv2D(filters=numberFilters, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING,
                           kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            input = Activation(config.RELU_FUNCTION) (input)
            input = MaxPooling2D(pool_size=(2,2), strides=2) (input)
            input = BatchNormalization() (input)
            input = Dropout(dropoutRate) (input)

            return input

        except:
            raise

    def build(self, *args, trainedModel=None) -> Sequential:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR THE INITIALIZATION OF SEQUENTIAL ALEXNET MODEL
        :param args: list integers, in logical order --> to populate cnn (filters) and dense (neurons)
        :return: Sequential: AlexNet MODEL
        '''

        try:

            #IF USER ALREADY HAVE A TRAINED MODEL, AND NO WANTS TO BUILD AGAIN A NEW MODEL
            if trainedModel != None:
                return trainedModel

            input_shape = (config.WIDTH, config.HEIGHT, config.CHANNELS)
            input = Input(input_shape)

            ## first convolution layer
            model = self.add_conv(input, args[2], 0.25, input_shape=input_shape)

            ## simple convolution layers
            numberFilters = args[2] + args[3]
            for i in range(args[0]):
                model = self.add_conv(model, numberFilters, 0.25)
                numberFilters += args[3]

            ## stacked convolutional layers
            for i in range(args[1]):
                model = self.add_stack(model, numberFilters, 0.25)
                numberFilters += args[3]

            ## flatten
            model = Flatten() (model)

            # Full Connected Layer(s)
            for i in range(args[4]):
                model = Dense(units=args[5], kernel_regularizer=regularizers.l2(config.DECAY)) (model)
                model = Activation(config.RELU_FUNCTION) (model)
                model = BatchNormalization() (model)
                if i != (args[4] - 1):
                    model = Dropout(0.25) (model) ## applies Dropout on all FCL's except FCL preceding the ouput layer (softmax)

            # Output Layer
            model = Dense(units=config.NUMBER_CLASSES) (model)
            model = Activation(config.SOFTMAX_FUNCTION) (model)

            ## model creation
            model = mp(inputs=input, outputs=model)

            if config.BUILD_SUMMARY == 1:
                model.summary()

            return model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_BUILD)

    def train(self, model : Sequential, *args) -> Tuple[History, Sequential]:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR MAKE THE TRAINING OF MODEL
        :param model: Sequential model builded before, or passed (already trained model)
        :param args: only one value batch size
        :return: Sequential model --> trained model
        :return: History.history --> train and validation loss and metrics variation along epochs
        '''

        try:

            if model is None:
                raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)

            # OPTIMIZER
            opt = Adam(learning_rate=config.LEARNING_RATE, decay=config.DECAY)

            # COMPILE
            model.compile(optimizer=opt, loss=config.LOSS_CATEGORICAL, metrics=[config.ACCURACY_METRIC])

            #GET STRATEGIES RETURN DATA, AND IF DATA_AUGMENTATION IS APPLIED TRAIN GENERATOR
            train_generator = None

            if len(self.StrategyList) == 0: #IF USER DOESN'T PRETEND EITHER UNDERSAMPLING AND OVERSAMPLING
                X_train = self.data.X_train
                y_train = self.data.y_train

            else: #USER WANTS AT LEAST UNDERSAMPLING OR OVERSAMPLING
                X_train, y_train = self.StrategyList[0].applyStrategy(self.data)
                if len(self.StrategyList) > 1: #USER CHOOSE DATA AUGMENTATION OPTION
                    train_generator = self.StrategyList[1].applyStrategy(self.data)

            es_callback = EarlyStopping(monitor='val_loss', patience=3)
            decrease_callback = ReduceLROnPlateau(monitor='val_loss',
                                                        patience=1,
                                                        factor=0.7,
                                                        mode='min',
                                                        verbose=1,
                                                        min_lr=0.000001)
            decrease_callback2 = ReduceLROnPlateau(monitor='val_loss',
                                                        patience=1,
                                                        factor=0.7,
                                                        mode='min',
                                                        verbose=1,
                                                        min_lr=0.000001)

            #CLASS WEIGHTS
            weights_y_train = config_func.decode_array(y_train)
            class_weights = class_weight.compute_class_weight('balanced',
                                                              numpy.unique(weights_y_train),
                                                              weights_y_train)

            if train_generator is None: #NO DATA AUGMENTATION

                history = model.fit(
                    x=X_train,
                    y=y_train,
                    batch_size=args[0],
                    epochs=config.EPOCHS,
                    validation_data=(self.data.X_val, self.data.y_val),
                    shuffle=True,
                    callbacks=[es_callback, decrease_callback, decrease_callback2],
                    class_weight=class_weights,
                    verbose=config.TRAIN_VERBOSE
                )

                return history, model

            #ELSE APPLY DATA AUGMENTATION

            history = model.fit_generator(
                generator=train_generator,
                validation_data=(self.data.X_val, self.data.y_val),
                epochs=config.EPOCHS,
                steps_per_epoch=X_train.shape[0] / args[0],
                shuffle=True,
                class_weight=class_weights,
                callbacks= [es_callback, decrease_callback, decrease_callback2],
                verbose=config.TRAIN_VERBOSE
            )

            return history, model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_TRAINING)

    def __str__(self):
        pass