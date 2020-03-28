from models import Model
import config
from exceptions import CustomError
from . import PSO, GA, Optimizer

class OptimizerFactory:

    def __init__(self):
        pass

    def createOptimizer(self, typeOptimizer : str, model : Model, *args) -> Optimizer:

        '''
        THIS FUNCTION IS USED TO CREATE INHERITED INSTANCES OF OPTIMIZERS, e.g PSO or GA
        :param typeOptimizer: str --> type of optimizer user wants
        :param model: Model Object --> model to associate with optimizer
        :param args: list of data --> (number individuals, iterations, dimensions of problem)
        :return:
        '''

        try:

            if model is None:
                raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)

            if typeOptimizer == config.PSO_OPTIMIZER:
                return PSO.PSO(model, *args)
            elif typeOptimizer == config.GA_OPTIMIZER:
                return GA.GA(model, *args)
            else:
                raise CustomError.ErrorCreationModel(config.ERROR_INVALID_OPTIMIZER)

        except:
            raise