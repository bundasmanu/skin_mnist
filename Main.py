from exceptions import CustomError
from models import AlexNet, VGGNet, Model, ModelFactory
from models.Strategies_Train import DataAugmentation, Strategy, UnderSampling, OverSampling
from optimizers import GA, PSO, Optimizer, OptimizerFactory
import pandas as pd
import config
import config_func
import os

def main():

    ROOT_DIR = os.getcwd()
    print(ROOT_DIR)
    INPUT_DIR = os.path.join(ROOT_DIR, config.INPUT_FOLDER)
    print(INPUT_DIR)
    PATIENTS_INFO = os.path.join(INPUT_DIR, config.INFO_PATIENTS)
    print(PATIENTS_INFO)

    IMAGES_REGEX = os.path.join(INPUT_DIR, config.IMAGES_ACESS)
    images_paths = config_func.getImages(IMAGES_REGEX)
    print(images_paths[:5])

    data = pd.read_csv(PATIENTS_INFO)
    print(data.iloc[0])
    data = data.sort_values(config.IMAGE_ID, ascending=True)
    print(data.head(5))

    #ADD NEW COLUMN (PATH IMAGE) AND POPULATE WITH COHERENT PATH FOR EACH IMAGE
    data = config_func.addNewColumn_Populate_DataFrame(data, config.PATH, images_paths)
    data = data.sort_index()
    print(data.head(5))
    print(data.iloc[0][config.PATH])

if __name__ == "__main__":
    main()