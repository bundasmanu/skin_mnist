from . import Model
import Data
from .Strategies_Train import Strategy
from exceptions import CustomError
import config
import config_func
import numpy
from keras.models import Model as mp, Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Input, BatchNormalization, Dense, Flatten, Add, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.callbacks.callbacks import History, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.initializers import glorot_uniform, he_uniform
from keras.regularizers import l2
from keras.utils import plot_model
from sklearn.utils import class_weight
from typing import Tuple
from models.Strategies_Train import DataAugmentation, UnderSampling, OverSampling

class ResNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(ResNet, self).__init__(data, *args)
    
    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        super(ResNet, self).addStrategy(strategy=strategy)

    def identity_block(self, tensor_input, *args):

        '''
        THIS FUNCTION SIMULES THE CONCEPT OF A IDENTITY BLOCK
            paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
        :param tensor_input: input_tensor result of previous block application on cnn architecture (conv_block or identity_block)
        :param args: number of filters to populate conv2d layers
        :return: tensor merge of input and identity conv blocks
        '''

        try:

            ## save copy input, because i need to apply alteration on tensor_input parameter, and in final i need to merge this two tensors
            input = tensor_input

            tensor_input = Conv2D(filters=args[0], padding=config.SAME_PADDING, kernel_size=(3,3), strides=1,
                                  kernel_initializer=he_uniform(config.HE_SEED), kernel_regularizer=l2(config.DECAY))(tensor_input)
            tensor_input = BatchNormalization(axis=3) (tensor_input) ## perform batch normalization alongside channels axis [samples, width, height, channels]
            tensor_input = Activation(config.RELU_FUNCTION) (tensor_input)

            tensor_input = Conv2D(filters=args[0], padding=config.SAME_PADDING, kernel_size=(3,3), strides=1,
                                  kernel_initializer=he_uniform(config.HE_SEED), kernel_regularizer=l2(config.DECAY))(tensor_input)
            tensor_input = BatchNormalization(axis=3) (tensor_input) ## perform batch normalization alongside channels axis [samples, width, height, channels]
            #tensor_input = Activation(config.RELU_FUNCTION) (tensor_input)

            ## now i need to merge initial input and identity block created, this is passed to activation function
            tensor_input = Add() ([tensor_input, input])
            tensor_input = Activation(config.RELU_FUNCTION) (tensor_input)

            return tensor_input

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_IDENTITY_BLOCK)

    def convolution_block(self, tensor_input, *args):

        '''
        THIS FUNCTIONS REPRESENTS THE CONCEPT OF CONVOLUTION BLOCK ON RESNET, COMBINING MAIN PATH AND SHORTCUT
            paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
        :param tensor_input: input_tensor result of previous block application on cnn architecture (conv_block or identity_block)
        :param args: number of filters to populate conv2d layers
        :return: tensor merge of path created using convs and final shortcut
        '''

        try:

            ## save copy input, because i need to apply alteration on tensor_input parameter, and in final i need to merge this two tensors
            shortcut_path = tensor_input

            tensor_input = Conv2D(filters=args[0], padding=config.SAME_PADDING, kernel_size=(3,3), strides=args[1], #RETRIEVE SOME SPACE
                                  kernel_initializer=he_uniform(config.HE_SEED), kernel_regularizer=l2(config.DECAY))(tensor_input)
            tensor_input = BatchNormalization(axis=3) (tensor_input) ## perform batch normalization alongside channels axis [samples, width, height, channels]
            tensor_input = Activation(config.RELU_FUNCTION) (tensor_input)

            tensor_input = Conv2D(filters=args[0], padding=config.SAME_PADDING, kernel_size=(3,3), strides=1,
                                  kernel_initializer=he_uniform(config.HE_SEED),kernel_regularizer=l2(config.DECAY))(tensor_input)
            tensor_input = BatchNormalization(axis=3) (tensor_input) ## perform batch normalization alongside channels axis [samples, width, height, channels]
            tensor_input = Activation(config.RELU_FUNCTION) (tensor_input)

            ## definition of shortcut path
            shortcut_path = Conv2D(filters=args[0], kernel_size=(1,1), strides=args[1], padding=config.SAME_PADDING,
                                   kernel_initializer=he_uniform(config.HE_SEED), kernel_regularizer=l2(config.DECAY)) (shortcut_path)
            shortcut_path = BatchNormalization(axis=3) (shortcut_path)

            ## now i need to merge conv path and shortcut path, this is passed to activation function
            tensor_input = Add() ([tensor_input, shortcut_path])
            tensor_input = Activation(config.RELU_FUNCTION) (tensor_input)

            return tensor_input

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_CONV_BLOCK)

    def build(self, *args, trainedModel=None) -> Sequential:

        # resnet v1, based on: https://keras.io/examples/cifar10_resnet/
        # ----------PAPER------------- https://arxiv.org/pdf/1512.03385.pdf
        # ---------RESNET 18 AND 34 ARCHITECTURE: https://datascience.stackexchange.com/questions/33022/how-to-interpert-resnet50-layer-types/47489
        # ---------VERY GOOD EXPLANATION: http://ethen8181.github.io/machine-learning/keras/resnet_cam/resnet_cam.html#Identity-Block
        ## model based on resnet-18 approach and described in paper cited in identity_block and convolution_block functions
        ## INFO THAT RESNET DOESN'T USE FULLY CONNECTED LAYERS (ONLY OUTPUT LAYER) --> https://stackoverflow.com/questions/44925467/does-resnet-have-fully-connected-layers
        try:

            # IF USER ALREADY HAVE A TRAINED MODEL, AND NO WANTS TO BUILD AGAIN A NEW MODEL
            if trainedModel != None:
                return trainedModel

            input_shape = (config.HEIGHT, config.WIDTH, config.CHANNELS)
            input_shape = Input(input_shape)

            X = ZeroPadding2D((3, 3))(input_shape)

            ## normal convolution layer --> first entry
            X = Conv2D(filters=args[0], kernel_size=(5, 5), strides=2, padding=config.SAME_PADDING,
                       kernel_initializer=he_uniform(config.HE_SEED), kernel_regularizer=l2(config.DECAY))(X)
            X = BatchNormalization(axis=3)(X)
            X = Activation(config.RELU_FUNCTION)(X)
            X = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(X)

            ## loop of convolution and identity blocks
            numberFilters = args[0]
            for i in range(args[1]):
                if i == 0:
                    X = self.convolution_block(X, *(numberFilters, 1)) #first set of building blocks, stride is 1
                else:
                    X = self.convolution_block(X, *(numberFilters, 2)) #next set of building blocks, stride is 2
                for i in range(args[2]):
                    X = self.identity_block(X, *(numberFilters, ))
                numberFilters += args[3]

            X = GlobalAveragePooling2D()(X)

            X = Dense(units=config.NUMBER_CLASSES, kernel_initializer=he_uniform(config.HE_SEED),
                      kernel_regularizer=l2(config.DECAY))(X)
            X = Activation(config.SOFTMAX_FUNCTION)(X)

            ## finally model creation
            model = mp(inputs=input_shape, outputs=X)

            if config.BUILD_SUMMARY == 1:
                model.summary()
            # plot_model(model, show_shapes=True, to_file='residual_module.png')

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

            # get data
            X_train = self.data.X_train
            y_train = self.data.y_train

            if self.StrategyList: # if strategylist is not empty
                for i, j in zip(self.StrategyList, range(len(self.StrategyList))):
                    if isinstance(i, DataAugmentation.DataAugmentation):
                        train_generator = self.StrategyList[j].applyStrategy(self.data)
                    if isinstance(i, OverSampling.OverSampling):
                        X_train, y_train = self.StrategyList[j].applyStrategy(self.data)
                    if isinstance(i, UnderSampling.UnderSampling):
                        X_train, y_train = self.StrategyList[j].applyStrategy(self.data)

            #CLASS WEIGHTS --> get train distributions before the application of undersampling
            weights_y_train = config_func.decode_array(y_train)
            class_weights = class_weight.compute_class_weight('balanced',
                                                               numpy.unique(weights_y_train),
                                                               weights_y_train)

            es_callback = EarlyStopping(monitor='val_loss', patience=2)
            decrease_callback = ReduceLROnPlateau(monitor='loss',
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
                verbose=config.TRAIN_VERBOSE,
                callbacks= [es_callback, decrease_callback, decrease_callback2]
            )

            return history, model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_TRAINING)

    def __str__(self):
        super(ResNet, self).__str__()