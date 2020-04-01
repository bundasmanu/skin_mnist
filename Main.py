from exceptions import CustomError
from models import AlexNet, VGGNet, Model, ModelFactory
from models.Strategies_Train import DataAugmentation, Strategy, UnderSampling, OverSampling
from optimizers import GA, PSO, Optimizer, OptimizerFactory
import pandas as pd
import config
import config_func
import os
from sklearn.model_selection import train_test_split
import Data

def main():

    print("\n###############################################################")
    print("##########################DATA PREPARATION#####################")
    print("###############################################################\n")
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

    #IMPUTATE NULL VALUES
    data = config_func.impute_null_values(data, config.AGE, mean=True)
    print(data.isnull().sum())
    print(data.head(5))
    data.dx = data.dx.astype('category')
    print(data.info())

    # GET IMAGE DATASET X (RGB VALUES) AND Y (TARGETS)
    # X, Y = config_func.getDataFromImages(dataframe=data, size=config.WANTED_IMAGES)
    # print(X.shape)
    # print(Y.shape)

    #GET IMAGE DATASET WITH SPECIFIC SIZE
    X, Y = config_func.getDataFromImages(dataframe=data, size=2000)
    print(X.shape)
    print(Y.shape)
    number_by_perc = [sum(Y == i) for i in range(len(data.dx.unique()))]

    # STRATIFY X_TEST, X_VAL AND X_TEST
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=config.VALIDATION_SPLIT, shuffle=True, stratify=Y,
                                                      random_state=config.RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=config.TEST_SPLIT, shuffle=True,
                                                        stratify=y_train, random_state=config.RANDOM_STATE)

    # NORMALIZE DATA
    X_train, X_val, X_test = config_func.normalize(X_train, X_val, X_test)

    # ONE HOT ENCODING TARGETS
    y_train, y_val, y_test = config_func.one_hot_encoding(y_train, y_val, y_test)

    # CREATION OF DATA OBJECT
    data_obj = Data.Data(X_train=X_train, X_val=X_val, X_test=X_test,
                         y_train=y_train, y_val=y_val, y_test=y_test)

    print("\n###############################################################")
    print("##########################CLASSIFICATION#######################")
    print("###############################################################\n")

if __name__ == "__main__":
    main()