from . import Optimizer
from models import Model
from deap import base, creator, tools, algorithms
from bitstring import BitArray
from scipy.stats import bernoulli
import config
from exceptions import CustomError
import config_func

class GA(Optimizer.Optimizer):

    def __init__(self, model : Model.Model, *args):
        super(GA, self).__init__(model, *args)

    def objectiveFunction(self, dimValues):

        '''
        THIS REPRESENTS OBJECTIVE FUNCTION OF GENETIC ALGORITHM --> This function is applied by all individuals in all generations
        :param dimValues: BitArray --> Bit Array with a dimension
        :return: cost: float --> represents individual loss
        '''

        try:

            #CONVERT BINARY VALUES TO INTEGER --> USING BITSTRING LIBRARY
            #USER CAN IMPLEMENT HERE, OR CAN CALL PARENT OBJECTIVE FUNCTION

            return 0.0
        except:
            raise


    def optimize(self):

        '''

        :return:
        '''

        try:

            creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
            creator.create('Individual', list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()
            toolbox.register('binary', bernoulli.rvs, 0.5)
            toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=self.dims) #REGISTER INDIVIDUAL
            toolbox.register('population', tools.initRepeat, list, toolbox.individual) #REGISTER POPULATION

            toolbox.register('mate', tools.cxOrdered) #CROSSOVER TECHNIQUE --> https://www.researchgate.net/figure/The-order-based-crossover-OX-a-and-the-insertion-mutation-b-operators_fig2_224330103
            toolbox.register('mutate', tools.mutShuffleIndexes, indpb=config.INDPB) #MUTATION TECHNIQUE --> https://www.mdpi.com/1999-4893/12/10/201/htm
            toolbox.register('select', tools.selTournament, tournsize=config.TOURNAMENT_SIZE) #IN MINIMIZATION PROBLEMS I CAN'T USE ROULETTE
            toolbox.register('evaluate', self.objectiveFunction) #EVALUATION FUNCTION

            population = toolbox.population(n=self.indiv)
            r = algorithms.eaSimple(population, toolbox, cxpb=config.CXPB, mutpb=config.MUTPB, ngen=self.iters, verbose=True)

            bestValue = tools.selBest(population, k=1) #I ONLY NEED BEST INDIVIDUAL --> ARRAY BIDIMENSIONAL (K=1, GENE_LENGTH)

            return bestValue
        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)