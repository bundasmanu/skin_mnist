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
#os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
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
                                                      random_state=config.RANDOM_STATE, stratify=Y)
    indexes = indeces_train
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_train, y_train, indexes, test_size=config.TEST_SPLIT,
                                                        shuffle=True, random_state=config.RANDOM_STATE, stratify=y_train)

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
    # definition of args to pass to template_method (conv's number of filters, dense neurons and batch size)
    alex_args = (
        2, # number of normal convolutional layer (+init conv)
        2, # number of stack cnn layers
        16, # number of feature maps of initial conv layer
        16, # growth rate
        1, # number of FCL Layers
        16, # number neurons of Full Connected Layer
        config.BATCH_SIZE_ALEX_AUG # batch size
    )

    # APPLY BUILD, TRAIN AND PREDICT
    #model, predictions, history = alexNet.template_method(*alex_args)

    ## PLOT FINAL RESULTS
    #config_func.print_final_results(data_obj.y_test, predictions, history, dict=False)

    ## ---------------------------VGGNET APPLICATION ------------------------------------

    ## DEFINITION OF NUMBER OF CNN AND DENSE LAYERS
    vggLayers = (5, 1)

    ## GET VGGNET MODEL
    vggnet = model_fact.getModel(config.VGG_NET, data_obj, *vggLayers)

    ## ATTRIBUTION OS TRAIN STRATEGIES
    vggnet.addStrategy(oversampling)
    vggnet.addStrategy(data_augment)

    # VALUES TO POPULATE ON CONV AND DENSE LAYERS
    vgg_args = (
        5,  # number of stack cnn layers (+ init stack)
        32,  # number of feature maps of initial conv layer
        12,  # growth rate
        1, # number of FCL Layers
        16,  # number neurons of Full Connected Layer
        config.BATCH_SIZE_ALEX_AUG  # batch size
    )

    # APPLY BUILD, TRAIN AND PREDICT
    #model, predictions, history = vggnet.template_method(*vgg_args)
    #vggnet.save(model, config.VGG_NET_WEIGHTS_FILE)

    ## PLOT FINAL RESULTS
    #config_func.print_final_results(data_obj.y_test, predictions, history, dict=False)

    ## ---------------------------RESNET APPLICATION ------------------------------------

    # number of conv and dense layers respectively
    number_cnn_dense = (5, 1)

    # creation of ResNet instance
    resnet = model_fact.getModel(config.RES_NET, data_obj, *number_cnn_dense)

    # apply strategies to resnet
    resnet.addStrategy(oversampling)
    resnet.addStrategy(data_augment)

    # definition of args to pass to template_method (conv's number of filters, dense neurons and batch size)
    resnet_args = (
        16,  # number of filters of initial CNN layer
        4,  # number of consecutive conv+identity blocks
        1, # number of identity block in each (conv+identity) block
        16,  # growth rate
        config.BATCH_SIZE_ALEX_AUG,  # batch size
    )

    # APPLY BUILD, TRAIN AND PREDICT
    #model, predictions, history = resnet.template_method(*resnet_args)
    #resnet.save(model, config.RES_NET_WEIGHTS_FILE)

    ## PLOT FINAL RESULTS
    #config_func.print_final_results(data_obj.y_test, predictions, history, dict=False)

    ## ---------------------------DENSENET APPLICATION ------------------------------------

    # # DICTIONARIES DEFINITION
    numberLayers = (
        4,  # BLOCKS
        1  # DENSE LAYERS
    )

    valuesLayers = (
        24,  # initial number of Feature Maps
        4,  # number of dense blocks
        5,  # number of layers in each block
        12,  # growth rate
        0.5,  # compression rate
        config.BATCH_SIZE_ALEX_AUG  # batch size
    )

    densenet = model_fact.getModel(config.DENSE_NET, data_obj, *numberLayers)

    densenet.addStrategy(oversampling)
    densenet.addStrategy(data_augment)

    model, predictions, history = densenet.template_method(*valuesLayers)

    config_func.print_final_results(data_obj.y_test, predictions, history)

    ## --------------------------- ENSEMBLE OF MODELS ------------------------------------

    # get weights of all methods from files
    # alexNet = load_model(config.ALEX_NET_WEIGHTS_FILE)
    # vggnet = load_model(config.VGG_NET_WEIGHTS_FILE)
    # resnet = load_model(config.RES_NET_WEIGHTS_FILE)
    #
    # models = [alexNet, vggnet, resnet]
    #
    # ##call ensemble method
    # ensemble_model = config_func.ensemble(models=models)
    # predictions = ensemble_model.predict(data_obj.X_test)
    # argmax_preds = np.argmax(predictions, axis=1)  # BY ROW, BY EACH SAMPLE
    # argmax_preds = keras.utils.to_categorical(argmax_preds)
    #
    # ## print final results
    # config_func.print_final_results(data_obj.y_test, argmax_preds, history=None, dict=False)

    ## --------------------------- PSO ------------------------------------------------

    # optimizer fabric object
    # opt_fact = OptimizerFactory.OptimizerFactory()
    #
    # # definition models optimizers
    # pso_alex = opt_fact.createOptimizer(config.PSO_OPTIMIZER, alexNet, *config.pso_init_args_alex)
    # pso_vgg = opt_fact.createOptimizer(config.PSO_OPTIMIZER, vggnet, *config.pso_init_args_vgg)
    # pso_resnet = opt_fact.createOptimizer(config.PSO_OPTIMIZER, resnet, *config.pso_init_args_resnet)
    # pso_densenet = opt_fact.createOptimizer(config.PSO_OPTIMIZER, densenet, *config.pso_init_args_densenet)
    #
    # # optimize and print best cost
    # cost, pos, optimizer = pso_vgg.optimize()
    # print("Custo: {}".format(cost))
    # config_func.print_Best_Position_PSO(pos, config.VGG_NET) # print position
    # pso_vgg.plotCostHistory(optimizer)
    # pso_vgg.plotPositionHistory(optimizer, np.array(config.X_LIMITS), np.array(config.Y_LIMITS), config.POS_VAR_EXP,
    #                            config.LABEL_X_AXIS, config.LABEL_Y_AXIS)

if __name__ == "__main__":
    main()