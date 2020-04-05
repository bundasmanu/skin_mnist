from exceptions import CustomError
from models import AlexNet, VGGNet, Model, ModelFactory
from models.Strategies_Train import DataAugmentation, Strategy, UnderSampling, OverSampling
from optimizers import GA, PSO, Optimizer, OptimizerFactory
import pandas as pd
import config
import config_func
from sklearn.model_selection import train_test_split
import Data
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #THIS LINE DISABLES GPU OPTIMIZATION

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
    X, Y = config_func.getDataFromImages(dataframe=data, size=config.WANTED_IMAGES)
    print(X.shape)
    print(Y.shape)
    #number_by_perc = [sum(Y == i) for i in range(len(data.dx.unique()))]

    # STRATIFY X_TEST, X_VAL AND X_TEST
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=config.VALIDATION_SPLIT, shuffle=True,
                                                      random_state=config.RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=config.TEST_SPLIT, shuffle=True,
                                                        random_state=config.RANDOM_STATE)

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)

    # NORMALIZE DATA
    X_train, X_val, X_test = config_func.normalize(X_train, X_val, X_test)

    # ONE HOT ENCODING TARGETS
    y_train, y_val, y_test = config_func.one_hot_encoding(y_train, y_val, y_test)

    print("\n###############################################################")
    print("##########################CLASSIFICATION#######################")
    print("###############################################################\n")

    # CREATION OF DATA OBJECT
    data_obj = Data.Data(X_train=X_train, X_val=X_val, X_test=X_test,
                         y_train=y_train, y_val=y_val, y_test=y_test)

    ## DEFINITION OF NUMBER OF CNN AND DENSE LAYERS
    args = (6,1)

    # CREATE MODEL FACTORY
    model_fact = ModelFactory.ModelFactory()
    alexNet = model_fact.getModel(config.ALEX_NET, data_obj, *args)

    # APPLY STRATEGIES OF TRAIN
    oversampling = OverSampling.OverSampling()
    data_augment = DataAugmentation.DataAugmentation()
    alexNet.addStrategy(oversampling)
    alexNet.addStrategy(data_augment)

    # VALUES TO POPULATE ON CONV AND DENSE LAYERS
    filters_cnn = (16, 16, 24, 32, 64, 96)
    dense_neurons = (200, )

    # APPLY BUILD, TRAIN AND PREDICT
    #model, predictions, history = alexNet.template_method(*(filters_cnn+dense_neurons))

    ## ---------------------------RESNET APPLICATION ------------------------------------

    ## definition number cnn and dense layers of resnet
    number_cnn_dense = (9 ,0)

    ## definition filters of resnet
    initial_conv = (8,)
    conv2_stage = (16, 24)
    conv3_stage = (32, 48)
    conv4_stage = (64, 72)
    conv5_stage = (96, 128)
    resnet_args = (
        initial_conv + conv2_stage + conv3_stage +
        conv4_stage + conv5_stage
    )

    resnet = model_fact.getModel(config.RES_NET, data_obj, *number_cnn_dense)
    resnet.addStrategy(oversampling)
    resnet.addStrategy(data_augment)

    model, predictions, history = resnet.template_method(*resnet_args)

    print(config_func.plot_cost_history(history))
    print(config_func.plot_accuracy_plot(history))
    predictions = config_func.decode_array(predictions) #DECODE ONE-HOT ENCODING PREDICTIONS ARRAY
    y_test_decoded = config_func.decode_array(resnet.data.y_test)  # DECODE ONE-HOT ENCODING y_test ARRAY
    report, confusion_mat = config_func.getConfusionMatrix(predictions, y_test_decoded)
    print(report)
    plt.figure()
    config_func.plot_confusion_matrix(confusion_mat, config.DICT_TARGETS)

if __name__ == "__main__":
    main()