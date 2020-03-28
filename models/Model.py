from abc import ABC, abstractmethod
from keras.models import Sequential
from typing import Tuple, List
from .Strategies_Train import OverSampling, UnderSampling, DataAugmentation, Strategy
from keras.callbacks.callbacks import History
import config
from exceptions import CustomError
import numpy as np
import keras
import Data
import matplotlib.pyplot as plt

class Model(ABC):

    StrategyList = list()

    @abstractmethod
    def __init__(self, data : Data.Data, numberCNNLayers, numberDenseLayers):
        self.nCNNLayers = numberCNNLayers
        self.nDenseLayers = numberDenseLayers
        self.data = data

    @abstractmethod
    def addStrategy(self, strategy : Strategy.Strategy) -> bool:

        '''
        THIS FUNCTION ESTABILISHES TRAINING STRATEGIES (UNDER SAMPLING AND OVER SAMPLING ARE INDEPENDENT, USER ONLY ACCEPTS ONE)
        IF USER WANTS TO USE STRATEGIES, NEED TO ADD STRATEGIES BEFORE CALL TEMPLATE FUNCTION (template_method)
        :param Strategy object --> inherited object descendent from Strategy, e.g UnderSampling, OverSampling or Data Augmentation
        :return: boolean --> True no errors occured, False --> problem on the definition of any strategy
        '''

        try:

            self.StrategyList.append(strategy)

            return True
        except:
            raise CustomError.ErrorCreationModel(config.ERROR_APPEND_STRATEGY)

    def template_method(self, *args) -> Tuple[Sequential, np.array, History]:

        '''
        https://refactoring.guru/design-patterns/template-method/python/example
        THIS FUNCTION REPRESENTS A TEMPLATE PATTERN TO EXECUTE THE ALL SEQUENCE OF JOBS TO DO
        :param: args: list of integers in logical order to populate cnn and dense layers (filters and neurons)
        :return: Sequential: trained model
        :return: numpy array: model test data predictions
        :return History.history: history of trained model
        '''

        try:

            model = self.build(*args)
            history, model = self.train(model)
            predictions = self.predict(model)

            return model, predictions, history
        except:
            raise CustomError.ErrorCreationModel(config.ERROR_MODEL_EXECUTION)

    @abstractmethod
    def build(self, *args, trainedModel=None) -> Sequential: #I PUT TRAINED MODEL ARGUMENT AFTER ARGS BECAUSE NON REQUIRED ARGUMENTS NEED TO BE AFTER *ARGS
        pass

    @abstractmethod
    def train(self, model : Sequential) -> Tuple[History, Sequential]:
        pass

    def predict(self, model : Sequential):

        '''
        THIS FUNCTION IS USED IN MODEL PREDICTIONS
        :param model: Sequential model result from training
        :return: numpy array: predictions of X_test data in categorical way
        '''

        try:

            predictions = model.predict(
                x=self.data.X_test,
                use_multiprocessing=config.MULTIPROCESSING
            )

            #CHECK PREDICTIONS OUTPUT WITH REAL TARGETS
            argmax_preds = np.argmax(predictions, axis=1) #BY ROW, BY EACH SAMPLE

            #I APPLY ONE HOT ENCODING, IN ORDER TO FACILITATE COMPARISON BETWEEN Y_TEST AND PREDICTIONS
            argmax_preds = keras.utils.to_categorical(argmax_preds)

            return argmax_preds

        except:
            raise

    def save(self, model : Sequential, filename : str) -> bool:

        '''
        THIS FUNCTION IS USED TO SAVE A TRAINED MODEL --> This function may be called after train and predict of a model
        :param model: Sequential object --> Sequential model object return in train
        :param filename: str --> name of file, where you want to save model
        :return: bool --> True with save model to file, and False if occurs an error in save process
        '''

        try:

            model.save(filename)
            del model
            return True

        except:
            return False

    @abstractmethod
    def __str__(self):
        return "Model(nº CNN : ", self.nCNNLayers, " nº Dense: ", self.nDenseLayers