from . import Model, AlexNet, VGGNet
import config
import Data

class ModelFactory:

    def __init__(self):
        pass

    def getModel(self, modelType, data: Data.Data, *args) -> Model:

        '''

        :param modelType: str --> with model inherited type user wants --> AlexNet, VGGNet, etc
        :param data: Data object --> data associated with model Object
        :param args: list of integers --> number os cnn layers and dense layers
        :return:
        '''

        try:

            if modelType == config.ALEX_NET:
                return AlexNet.AlexNet(data, *args)
            elif modelType == config.VGG_NET:
                return VGGNet.VGGNet(data, *args)
            else:
                return AttributeError()

        except:
            raise