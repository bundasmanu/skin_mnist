from abc import ABC, abstractmethod
from models import Model
from exceptions import CustomError
import config

class Optimizer(ABC):

    def __init__(self, model : Model.Model, individuals, iterations, dimensions):
        if model == None:
            return CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)
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

        try:

            #OBJECTIVE FUNCTION NEED TO BE DEFINED ACORDING TO PROBLEM IN HANDS
            cnnFilters = [args[i] for i in range(self.model.nCNNLayers)] #ATTRIBUTION IMPORTANCE TO CNN FILTERS (*i) --> LAST FCONVOLUTION LAYER IS MORE IMPORTANT THAN FIRST
            totalMajorFilters = sum(cnnFilters[0:round(len(cnnFilters)/2)])
            totalMinorFilters = sum(cnnFilters[round(len(cnnFilters)/2):])
            denseNeurons = [args[(self.model.nCNNLayers+self.model.nDenseLayers) - (i+1)] for i in range(self.model.nDenseLayers)]
            if not denseNeurons: # if list is empty
                totalNeurons = 1 # resnet case --> doesn't have dense totalneurons doesn't count is multiplication is 0
            else:
                totalNeurons = sum(denseNeurons)

            # get report from args
            report = args[-1]

            ## https://stackoverflow.com/questions/48417867/access-to-numbers-in-classification-report-sklearn
            macro_precision = report['macro avg']['precision']
            macro_recall = report['macro avg']['recall']

            return 3.5 * (1.0 - (1.0 / (totalMajorFilters))) + 2.0 * (1.0 - (1.0 / (totalMinorFilters)))\
                          + 3.5 * (1.0 - (1.0 / (totalNeurons))) + 3.0 * (1.0 - acc) \
                            + 4.0 * (1.0 - macro_precision) + 5.0 * (1.0 - macro_recall)

        except:
            raise

    @abstractmethod
    def optimize(self):
        pass