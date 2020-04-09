from models import Model
import Data
from models.Strategies_Train import Strategy
from exceptions import CustomError
import config
import config_func
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import os
from keras.models import Model as mp
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, UpSampling2D, Conv2DTranspose, Activation, Reshape, Input, Dropout, concatenate, ConvLSTM2D
import numpy as np
import h5py

class UNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(UNet, self).__init__(data, *args)

    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        super(UNet, self).addStrategy(strategy)

    def build(self, *args, trainedModel=None):

        try:

            input_shape = (config.HEIGHT, config.WIDTH, config.CHANNELS)
            img_input = Input(input_shape)
            x = Conv2D(64, (3, 3), padding=config.SAME_PADDING, name='conv1', strides=(1, 1))(img_input)
            x = BatchNormalization(name='bn1')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2D(64, (3, 3), padding=config.SAME_PADDING, name='conv2')(x)
            x = BatchNormalization(name='bn2')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = MaxPooling2D()(x)

            x = Conv2D(128, (3, 3), padding=config.SAME_PADDING, name='conv3')(x)
            x = BatchNormalization(name='bn3')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2D(128, (3, 3), padding=config.SAME_PADDING, name='conv4')(x)
            x = BatchNormalization(name='bn4')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = MaxPooling2D()(x)

            x = Conv2D(256, (3, 3), padding=config.SAME_PADDING, name='conv5')(x)
            x = BatchNormalization(name='bn5')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2D(256, (3, 3), padding=config.SAME_PADDING, name='conv6')(x)
            x = BatchNormalization(name='bn6')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2D(256, (3, 3), padding=config.SAME_PADDING, name='conv7')(x)
            x = BatchNormalization(name='bn7')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = MaxPooling2D()(x)

            x = Conv2D(512, (3, 3), padding=config.SAME_PADDING, name='conv8')(x)
            x = BatchNormalization(name='bn8')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2D(512, (3, 3), padding=config.SAME_PADDING, name='conv9')(x)
            x = BatchNormalization(name='bn9')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2D(512, (3, 3), padding=config.SAME_PADDING, name='conv10')(x)
            x = BatchNormalization(name='bn10')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = MaxPooling2D()(x)

            x = Conv2D(512, (3, 3), padding=config.SAME_PADDING, name='conv11')(x)
            x = BatchNormalization(name='bn11')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2D(512, (3, 3), padding=config.SAME_PADDING, name='conv12')(x)
            x = BatchNormalization(name='bn12')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2D(512, (3, 3), padding=config.SAME_PADDING, name='conv13')(x)
            x = BatchNormalization(name='bn13')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = MaxPooling2D()(x)

            x = Dense(1024, activation=config.RELU_FUNCTION, name='fc1')(x)
            x = Dense(1024, activation=config.RELU_FUNCTION, name='fc2')(x)

            # Decoding Layer
            x = UpSampling2D()(x)
            x = Conv2DTranspose(512, (3, 3), padding=config.SAME_PADDING, name='deconv1')(x)
            x = BatchNormalization(name='bn14')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2DTranspose(512, (3, 3), padding=config.SAME_PADDING, name='deconv2')(x)
            x = BatchNormalization(name='bn15')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2DTranspose(512, (3, 3), padding=config.SAME_PADDING, name='deconv3')(x)
            x = BatchNormalization(name='bn16')(x)
            x = Activation(config.RELU_FUNCTION)(x)

            x = UpSampling2D()(x)
            x = Conv2DTranspose(512, (3, 3), padding=config.SAME_PADDING, name='deconv4')(x)
            x = BatchNormalization(name='bn17')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2DTranspose(512, (3, 3), padding=config.SAME_PADDING, name='deconv5')(x)
            x = BatchNormalization(name='bn18')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2DTranspose(256, (3, 3), padding=config.SAME_PADDING, name='deconv6')(x)
            x = BatchNormalization(name='bn19')(x)
            x = Activation(config.RELU_FUNCTION)(x)

            x = UpSampling2D()(x)
            x = Conv2DTranspose(256, (3, 3), padding=config.SAME_PADDING, name='deconv7')(x)
            x = BatchNormalization(name='bn20')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2DTranspose(256, (3, 3), padding=config.SAME_PADDING, name='deconv8')(x)
            x = BatchNormalization(name='bn21')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2DTranspose(128, (3, 3), padding=config.SAME_PADDING, name='deconv9')(x)
            x = BatchNormalization(name='bn22')(x)
            x = Activation(config.RELU_FUNCTION)(x)

            x = UpSampling2D()(x)
            x = Conv2DTranspose(128, (3, 3), padding=config.SAME_PADDING, name='deconv10')(x)
            x = BatchNormalization(name='bn23')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2DTranspose(64, (3, 3), padding=config.SAME_PADDING, name='deconv11')(x)
            x = BatchNormalization(name='bn24')(x)
            x = Activation(config.RELU_FUNCTION)(x)

            x = UpSampling2D()(x)
            x = Conv2DTranspose(64, (3, 3), padding=config.SAME_PADDING, name='deconv12')(x)
            x = BatchNormalization(name='bn25')(x)
            x = Activation(config.RELU_FUNCTION)(x)
            x = Conv2DTranspose(1, (3, 3), padding=config.SAME_PADDING, name='deconv13')(x)
            x = BatchNormalization(name='bn26')(x)
            x = Activation(config.SIGMOID_FUNCTION)(x)

            pred = Reshape((config.HEIGHT, config.WIDTH))(x) #reshape to single channel
            model = mp(inputs=img_input, outputs=pred)

            # input_size =(config.WIDTH, config.HEIGHT, config.CHANNELS)
            # N = input_size[0]
            # inputs = Input(input_size)
            # conv1 = Conv2D(64, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(inputs)
            # conv1 = Conv2D(64, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv1)
            #
            # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            # conv2 = Conv2D(128, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(pool1)
            # conv2 = Conv2D(128, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv2)
            # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            # conv3 = Conv2D(256, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(pool2)
            # conv3 = Conv2D(256, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv3)
            # drop3 = Dropout(0.5)(conv3)
            # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            # # D1
            # conv4 = Conv2D(512, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(pool3)
            # conv4_1 = Conv2D(512, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv4)
            # drop4_1 = Dropout(0.5)(conv4_1)
            # # D2
            # conv4_2 = Conv2D(512, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(drop4_1)
            # conv4_2 = Conv2D(512, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv4_2)
            # conv4_2 = Dropout(0.5)(conv4_2)
            # # D3
            # merge_dense = concatenate([conv4_2, drop4_1], axis=3)
            # conv4_3 = Conv2D(512, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(merge_dense)
            # conv4_3 = Conv2D(512, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv4_3)
            # drop4_3 = Dropout(0.5)(conv4_3)
            #
            # up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding=config.SAME_PADDING, kernel_initializer='he_normal')(
            #     drop4_3)
            # up6 = BatchNormalization(axis=3)(up6)
            # up6 = Activation(config.RELU_FUNCTION)(up6)
            #
            # x1 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(drop3)
            # x2 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(up6)
            # merge6 = concatenate([x1, x2], axis=1)
            # merge6 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding=config.SAME_PADDING, return_sequences=False,
            #                     go_backwards=True, kernel_initializer='he_normal')(merge6)
            #
            # conv6 = Conv2D(256, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(merge6)
            # conv6 = Conv2D(256, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv6)
            #
            # up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv6)
            # up7 = BatchNormalization(axis=3)(up7)
            # up7 = Activation(config.RELU_FUNCTION)(up7)
            #
            # x1 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(conv2)
            # x2 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(up7)
            # merge7 = concatenate([x1, x2], axis=1)
            # merge7 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding=config.SAME_PADDING, return_sequences=False,
            #                     go_backwards=True, kernel_initializer='he_normal')(merge7)
            #
            # conv7 = Conv2D(128, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(merge7)
            # conv7 = Conv2D(128, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv7)
            #
            # up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv7)
            # up8 = BatchNormalization(axis=3)(up8)
            # up8 = Activation(config.RELU_FUNCTION)(up8)
            #
            # x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
            # x2 = Reshape(target_shape=(1, N, N, 64))(up8)
            # merge8 = concatenate([x1, x2], axis=1)
            # merge8 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding=config.SAME_PADDING, return_sequences=False,
            #                     go_backwards=True, kernel_initializer='he_normal')(merge8)
            #
            # conv8 = Conv2D(64, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(merge8)
            # conv8 = Conv2D(64, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv8)
            # conv8 = Conv2D(2, 3, activation=config.RELU_FUNCTION, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv8)
            # conv9 = Conv2D(1, 1, activation=config.SIGMOID_FUNCTION)(conv8)
            #
            # model = mp(input=inputs, output=conv9)

            model.summary()
            return model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_UNET_STRATEGY)

    def train(self, model : Model):

        try:

            if model is None:
                raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)
                return None

            ## get weights file
            #file_unet_weights = os.path.join(os.getcwd(), config.UNET_WIGHTS_PATH)
            file_unet_weights = os.path.join(os.getcwd(), config.UNET_WIGHTS_PATH)

            model.load_weights(file_unet_weights)

            return [], model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_UNET_STRATEGY)

    def predict(self, model : Model):

        try:

            segmented_images = []
            for i in range(self.data.X_train.shape[0]):

                ## get copy of image, to avoid damage content
                image_copy = np.copy(self.data.X_train[i])

                ##reshape to 4 dimensions expected input (batch, width, height, channels)
                reshaped_image = image_copy.reshape(1, image_copy.shape[0],
                                                                     image_copy.shape[1], image_copy.shape[2])

                ## get predicted values for pixels on image from unet predict
                predicted_mask_values = model.predict(reshaped_image)
                predicted_mask_values =predicted_mask_values.reshape\
                    (predicted_mask_values.shape[0] * predicted_mask_values.shape[1], predicted_mask_values.shape[2])

                ## create binary mask with predicted values of image
                mask = config_func.defineMask(predicted_mask_values)

                ## concatenate real image and mask
                concatenated_mask = config_func.concate_image_mask(image_copy, mask)

                ## appen segmented image to list of predicted images
                segmented_images.append(concatenated_mask)

            return segmented_images

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_UNET_STRATEGY)

    def __str__(self):
        pass