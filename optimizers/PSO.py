from . import Optimizer
from models import Model
from exceptions import CustomError
import config
import pyswarms as ps
import numpy as np
from typing import Tuple
import math
from pyswarms.utils.plotters import plot_cost_history, plot_contour
import matplotlib.pyplot as plt
import config_func
from pyswarms.utils.plotters.formatters import Designer
from IPython.display import Image

class PSO(Optimizer.Optimizer):

    def __init__(self, model : Model.Model, *args): #DIMENSIONS NEED TO BE EQUAL TO NUMBER OF LAYERS ON MODEL
        super(PSO, self).__init__(model, *args)

    def plotCostHistory(self, optimizer):

        '''
        :param optimizer: optimizer object returned in the application/definition of PSO
        '''

        try:

            plot_cost_history(cost_history=optimizer.cost_history)

            plt.show()

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_PLOTTING)

    def plotPositionHistory(self, optimizer, xLimits, yLimits, filename, xLabel, yLabel):

        '''
        :param optimizer: optimizer object returned in the application/definition of PSO
        :param xLimits: numpy array (minLimit, maxLimit) of x Axis
        :param yLimits: numpy array (minLimit, maxLimit) of y Axis
        :param filename: name of filename returned by plot_contour (html gif)
        :param xLabel: name of X axis
        :param yLabel: name of Y axis
        '''

        try:

            d = Designer(limits=[xLimits, yLimits], label=[xLabel, yLabel])
            animation = plot_contour(pos_history=optimizer.pos_history,
                                     designer=d)

            animation.save(filename, writer='ffmpeg', fps=10)
            Image(url=filename)

            plt.show()
        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_PLOTTING)

    def boundsDefinition(self):

        '''
        This function has as main objective to define the limits of the dimensions of the problem
        :return: 2 numpy arrays --> shape(dimensionsProblem, ) with min and max values for each dimension of the problem
        '''

        try:

            ## EXAMPLE BOUNDS DEFINITION, USER CAN DEFINE OUR OWN BOUNDS
            totalDimensions = self.dims

            minBounds = np.ones(totalDimensions)
            maxBounds = np.ones(totalDimensions)

            ## all dimensions are in a range of [1-256]
            maxBounds = maxBounds * 64

            ## treat now only batch size bounds
            maxBounds[-1] = 40
            minBounds[-1] = 32

            bounds = (minBounds, maxBounds)

            return bounds

        except:
            raise

    def objectiveFunction(self, acc, *args):
        ## USER CAN DEFINE HERE OR CALL PARENT OBJECTIVE FUNCTION
        return super(PSO, self).objectiveFunction(acc, *args)

    def loopAllParticles(self, particles):

        '''
        THIS FUNCTION APPLIES PARTICLES ITERATION, EXECUTION CNN MODEL
        :param particles: numpy array of shape (nParticles, dimensions)
        :return: list: all losses returned along all particles iteration
        '''

        try:

            losses = []
            for i in range(particles.shape[0]):
                int_converted_values = [math.trunc(i) for i in particles[i]] #CONVERSION OF DIMENSION VALUES OF PARTICLE
                model, predictions, history = self.model.template_method(*int_converted_values) #APPLY BUILD, TRAIN AND PREDICT MODEL OPERATIONS, FOR EACH PARTICLE AND ITERATION
                decoded_predictions = config_func.decode_array(predictions)
                decoded_y_true = config_func.decode_array(self.model.data.y_test)
                report, conf = config_func.getConfusionMatrix(decoded_predictions, decoded_y_true, dict=True)
                acc = report['accuracy'] ## i can't compare y_test and predict, because some classes may have been unclassified
                int_converted_values.append(report)
                losses.append(self.objectiveFunction(acc, *int_converted_values)) #ADD COST LOSS TO LIST
            return losses

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)

    def optimize(self) -> Tuple[float, float, ps.general_optimizer.SwarmOptimizer]:

        '''
        THIS FUNCTION IS RESPONSIBLE TO APPLY ALL LOGIC OF PSO CNN NETWORK OPTIMIZATION
        :return: [float, float, SwarmOptimizer] --> best cost, best particle position and pso optimizer
        '''

        try:

            #DEFINITION OF BOUNDS
            bounds = self.boundsDefinition()

            optimizer = None
            if config.TOPOLOGY_FLAG == 0: #global best topology
                optimizer = ps.single.GlobalBestPSO(n_particles=self.indiv, dimensions=self.dims,
                                                    options=config.gbestOptions, bounds=bounds)
            else: #local best topology
                optimizer = ps.single.LocalBestPSO(n_particles=self.indiv, dimensions=self.dims,
                                                    options=config.lbestOptions, bounds=bounds)

            cost, pos = optimizer.optimize(objective_func=self.loopAllParticles, iters=self.iters)

            return cost, pos, optimizer

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)