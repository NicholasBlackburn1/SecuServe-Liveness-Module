"""
this is where the liveness model is for training and handling the creation of the liveness detection
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from utils import consoleLog


class LivenessModel(object):

    IMG_SIZE = 24


    #* this is for training the custom liveness detection
    def train(self, train_generator, val_generator):
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
        
        consoleLog.Warning("Starting to Train Model.....")
        model = Sequential()

        model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(self.IMG_SIZE,self.IMG_SIZE,1)))
        model.add(AveragePooling2D())

        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(AveragePooling2D())

        model.add(Flatten())

        model.add(Dense(units=120, activation='relu'))

        model.add(Dense(units=84, activation='relu'))

        model.add(Dense(units=1, activation = 'sigmoid'))


        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        print('[LOG] Training CNN')
    
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=val_generator,
                            validation_steps=STEP_SIZE_VALID,
                            epochs=20
        )
        consoleLog.PipeLine_Ok("Finished Training Model Successfull....")
        return model