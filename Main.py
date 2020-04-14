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
import cv2
from keras.models import load_model
import keras
import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #THIS LINE DISABLES GPU OPTIMIZATION

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

    #GET IMAGE DATASET WITH SPECIFIC SIZE
    X, Y = config_func.getDataFromImages(dataframe=data, size=config.WANTED_IMAGES)
    print(X.shape)
    print(Y.shape)
    #number_by_perc = [sum(Y == i) for i in range(len(data.dx.unique()))]

    # STRATIFY X_TEST, X_VAL AND X_TEST
    indexes = np.arange(X.shape[0])
    X_train, X_val, y_train, y_val, indeces_train, indices_val = train_test_split(X, Y, indexes, test_size=config.VALIDATION_SPLIT, shuffle=True,
                                                      random_state=config.RANDOM_STATE)
    indexes = indeces_train
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_train, y_train, indexes, test_size=config.TEST_SPLIT,
                                                        shuffle=True, random_state=config.RANDOM_STATE)

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)


    if config.FLAG_SEGMENT_IMAGES == 1:
        ## ---------------------------U-NET APPLICATION ------------------------------------
        dataset = Data.Data(X_train=X_train, X_val=X_val, X_test=X_test,
                         y_train=y_train, y_val=y_val, y_test=y_test)
        unet_args = (0, 0) # args doesn't matter --> any tuple is valid here, only in U-Net model

        fact = ModelFactory.ModelFactory()
        unet = fact.getModel(config.U_NET, dataset, *unet_args) # args doesn't matter

        ## check save and load predictions array to file
        PREDICTIONS_TEMP_FILE_PATH = os.path.join(INPUT_DIR, config.TEMP_ARRAYS)
        if os.path.exists(PREDICTIONS_TEMP_FILE_PATH):
            with open(PREDICTIONS_TEMP_FILE_PATH, 'rb') as f:
                predictions = np.load(f)
        else: ## if not exists
            with open(PREDICTIONS_TEMP_FILE_PATH, 'wb') as f:
                model, predictions, history = unet.template_method()
                predictions = np.array(predictions) ## transform list to numpy array
                np.save(f, predictions)

        ## create folder if not exists
        masks_path_folder = os.path.join(INPUT_DIR, config.MASKS_FOLDER)
        if not os.path.exists(masks_path_folder):
            os.makedirs(masks_path_folder)
        if not os.listdir(masks_path_folder): ## if folder is empty (no images inside)
            ## insert mask images in mask folder
            for i in range(predictions.shape[0]):
                cv2.imwrite(os.path.join(masks_path_folder, data.at[indices_train[i], config.IMAGE_ID]+'.jpg'), predictions[i])

        # plt.figure(figsize=(16, 16))
        # plt.imshow(cv2.cvtColor(self.data.X_train[2], cv2.COLOR_BGR2RGB))
        # plt.title('Original Image')
        # plt.show()
        # plt.imshow(mask, plt.cm.binary_r)
        # plt.title('Binary Mask')
        # plt.show()
        # plt.imshow(cv2.cvtColor(concatenated_mask, cv2.COLOR_BGR2RGB))
        # plt.title('Segmented Image')
        # plt.show()

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

    ## INSTANCE OF MODEL FACTORY
    model_fact = ModelFactory.ModelFactory()

    ## STRATEGIES OF TRAIN INSTANCES
    undersampling = UnderSampling.UnderSampling()
    oversampling = OverSampling.OverSampling()
    data_augment = DataAugmentation.DataAugmentation()

    ## ---------------------------ALEXNET APPLICATION ------------------------------------

    ## DEFINITION OF NUMBER OF CNN AND DENSE LAYERS
    args = (6,1)

    # CREATE MODEL
    alexNet = model_fact.getModel(config.ALEX_NET, data_obj, *args)

    # APPLY STRATEGIES OF TRAIN
    #alexNet.addStrategy(undersampling)
    alexNet.addStrategy(oversampling)
    alexNet.addStrategy(data_augment)

    # VALUES TO POPULATE ON CONV AND DENSE LAYERS
    filters_cnn = (96, 96, 96, 72, 64, 64)
    dense_neurons = (14, )
    batch_size = (config.BATCH_SIZE_ALEX_AUG, )

    # APPLY BUILD, TRAIN AND PREDICT
    #model, predictions, history = alexNet.template_method(*(filters_cnn+dense_neurons+batch_size))

    ## PLOT FINAL RESULTS
    #config_func.print_final_results(data_obj.y_test, predictions, history)

    ## ---------------------------VGGNET APPLICATION ------------------------------------

    ## DEFINITION OF NUMBER OF CNN AND DENSE LAYERS
    vggLayers = (5, 2)

    ## GET VGGNET MODEL
    vggnet = model_fact.getModel(config.VGG_NET, data_obj, *vggLayers)

    ## ATTRIBUTION OS TRAIN STRATEGIES
    vggnet.addStrategy(oversampling)
    vggnet.addStrategy(data_augment)

    # VALUES TO POPULATE ON CONV AND DENSE LAYERS
    filters_cnn = (16, 16, 32, 32, 48, 72)
    dense_neurons = (16, 8)
    batch_size = (config.BATCH_SIZE_ALEX_AUG, )

    # APPLY BUILD, TRAIN AND PREDICT
    #model, predictions, history = vggnet.template_method(*(filters_cnn+dense_neurons+batch_size))
    #vggnet.save(model, config.VGG_NET_WEIGHTS_FILE)

    ## PLOT FINAL RESULTS
    #config_func.print_final_results(data_obj.y_test, predictions, history)

    ## ---------------------------RESNET APPLICATION ------------------------------------

    ## definition number cnn and dense layers of resnet
    number_cnn_dense = (9 ,0)

    ## definition filters of resnet
    initial_conv = (72,)
    conv2_stage = (72, 84)
    conv3_stage = (84, 96)
    conv4_stage = (96, 128)
    conv5_stage = (128, 128)
    batch_size = (config.BATCH_SIZE_ALEX_AUG, )
    resnet_args = (
        initial_conv + conv2_stage + conv3_stage +
        conv4_stage + conv5_stage + batch_size
    )

    ## GET MODEL AND DEFINE STRATEGIES
    resnet = model_fact.getModel(config.RES_NET, data_obj, *number_cnn_dense)
    resnet.addStrategy(oversampling)
    resnet.addStrategy(data_augment)

    # APPLY BUILD, TRAIN AND PREDICT
    model, predictions, history = resnet.template_method(*resnet_args)
    resnet.save(model, config.RES_NET_WEIGHTS_FILE)

    ## PLOT FINAL RESULTS
    config_func.print_final_results(data_obj.y_test, predictions, history)

    ## --------------------------- ENSEMBLE OF MODELS ------------------------------------

    ## get weights of all methods from files
    # vggnet = load_model(config.VGG_NET_WEIGHTS_FILE)
    # resnet = load_model(config.RES_NET_WEIGHTS_FILE)
    #
    # models = [vggnet, resnet]
    #
    # ##call ensemble method
    # ensemble_model = config_func.ensemble(models=models)
    # predictions = ensemble_model.predict(data_obj.X_test)
    # argmax_preds = np.argmax(predictions, axis=1)  # BY ROW, BY EACH SAMPLE
    # argmax_preds = keras.utils.to_categorical(argmax_preds)
    #
    # ## print final results
    # config_func.print_final_results(data_obj.y_test, argmax_preds, history=None)

if __name__ == "__main__":
    main()