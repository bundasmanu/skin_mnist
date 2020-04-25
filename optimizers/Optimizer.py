from abc import ABC, abstractmethod
from models import Model
from exceptions import CustomError
import config
from keras import backend as K
import numpy as np

class Optimizer(ABC):

    def __init__(self, model : Model.Model, individuals, iterations, dimensions):
        if model == None:
            raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)
        self.model = model
        self.indiv = individuals
        self.iters = iterations
        self.dims = dimensions

    @abstractmethod
    def objectiveFunction(self, acc, *args):

        '''

        :param score: final score on train
        :param args: values of model layers --> the number of args need to be equal to total of layer used
                    e.g: args: (32, 48, 64, 32, 16) --> the sum of nCNNLayers and nDenseLayers need to be equal to number of args
                    last argument is metrics report
        :return: lost function
        '''

        '''
        :param score: final score on train
        :param args: first argument is a Keras Model
                    last argument is a confusion matrix
        :return: lost function
        '''

        try:

            # get report
            report = args[-1]
            recall_idc = report['macro avg']['recall']
            precision_idc = report['macro avg']['precision']

            # get model
            model = args[0]
            trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])

            return 1e-15 * trainable_count + 2 * (1.0 - acc) \
                            + 4 * (1.0 - recall_idc) + 3 * (1.0 - precision_idc)

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)

    @abstractmethod
    def optimize(self):
        pass