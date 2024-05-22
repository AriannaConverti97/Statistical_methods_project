from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,AveragePooling2D,MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Input, BatchNormalization

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import ResNet50, VGG16

import keras_tuner as kt

class cnn_custom(kt.HyperModel):

    def __init__(self, img_size, channels):
        self.img_size= img_size
        self.channels = channels

    def build(self, hp):
        model = Sequential([
                Input(shape=(self.img_size, self.img_size, self.channels)),

                Conv2D(32, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),

                Conv2D(64, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),

                Flatten(),

                Dense(
                    units=hp.Int('units', min_value=32, max_value=128, step=32, default=64),
                    activation='relu'
                ),
                Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.5)),

                Dense(1, activation='sigmoid')
            
            ], name='cnn_custom')
        
        model.compile(
            optimizer=Adam(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def name(self):
        return self.build(kt.HyperParameters()).name
    
    def param(self):
        hp=kt.HyperParameters()
        self.build(hp)
        return hp
    
    def summary(self):
        print(self.build(kt.HyperParameters()).summary())
    
class leNet5(kt.HyperModel):
    def __init__(self, img_size, channels):
        self.img_size= img_size
        self.channels = channels

    def build(self, hp):
        model = Sequential([
                    Input(shape=(self.img_size, self.img_size, self.channels)),
                    
                    Conv2D(6, kernel_size=(5,5), activation='relu'),
                    AveragePooling2D(pool_size=(2,2)),
                    
                    Conv2D(16, kernel_size=(5, 5), activation='relu'),
                    AveragePooling2D(pool_size=(2,2)),
                    
                    Dense(120,'relu'),
                    Flatten(),
                    Dense(units=hp.Int('units', min_value=32, max_value=84, step=32, default=84),
                                activation='relu'),

                    Dense(1,'sigmoid')
                ], name='leNet5')
    
        model.compile(
                    loss='binary_crossentropy',
                    optimizer=Adam(
                            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
                    ),
                    metrics=['accuracy']
        )
        
        return model
    
    def name(self):
        return self.build(kt.HyperParameters()).name
    
    def param(self):
        hp=kt.HyperParameters()
        self.build(hp)
        return hp
    
    def summary(self):
        print(self.build(kt.HyperParameters()).summary())
    
    
class alexNet(kt.HyperModel): 
    def __init__(self, img_size, channels):
        self.img_size= img_size
        self.channels = channels

    def build(self, hp):
        model = Sequential([
                    Input(shape=(self.img_size, self.img_size, self.channels)),
                    
                    Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'),
                    MaxPooling2D(pool_size=3, strides=2),
                    
                    Conv2D(filters=256, kernel_size=5, padding='same',activation='relu'),
                    MaxPooling2D(pool_size=3, strides=2),
                    
                    Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
                    Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
                    Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
                    MaxPooling2D(pool_size=3, strides=2),
                    
                    Flatten(),
                    Dense(4096, activation='relu'),
                    Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.5)),
                    Dense(4096, activation='relu'),
                    Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1, default=0.5)),
           
                    Dense(1, 'sigmoid')
                ], name='alexNet')
    
        model.compile(
                    loss='binary_crossentropy',
                    optimizer=Adam(
                            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
                    ),
                    metrics=['accuracy']
        )
        
        return model
    
    def name(self):
        return self.build(kt.HyperParameters()).name
    
    def param(self):
        hp=kt.HyperParameters()
        self.build(hp)
        return hp
    
    def summary(self, hp):
        print(self.build(hp).summary())
    
class vgg16(kt.HyperModel):
    def __init__(self, img_size, channels):
        self.img_size= img_size
        self.channels = channels

    def build(self, hp):
        input = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(self.img_size, self.img_size, self.channels)))
        x = input.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=input.input, outputs=output, name='vgg16')

        for layer in input.layers:
            layer.trainable = False

       
        model.compile(
                    loss='binary_crossentropy',
                    optimizer=Adam(
                            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
                    ),
                    metrics=['accuracy']
        )
        
        return model
    
    def name(self):
        return self.build(kt.HyperParameters()).name
    
    def param(self):
        hp=kt.HyperParameters()
        self.build(hp)
        return hp
    def summary(self):
        print(self.build(kt.HyperParameters()).summary())
    
class resNet50(kt.HyperModel):
    def __init__(self, img_size, channels):
        self.img_size= img_size
        self.channels = channels

    def build(self, hp):
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE,IMG_SIZE,3)))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)


        model = tf.keras.Model(inputs=base_model.input, outputs=predictions, name='resNet50')

        for layer in base_model.layers:
            layer.trainable = False
    
        model.compile(
                    loss='binary_crossentropy',
                    optimizer=Adam(
                            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
                    ),
                    metrics=['accuracy']
        )
        
        return model
    
    def name(self):
        return self.build(kt.HyperParameters()).name
    
    def param(self):
        hp=kt.HyperParameters()
        self.build(hp)
        return hp
    def summary(self):
        print(self.build(kt.HyperParameters()).summary())
    
class mlp(kt.HyperModel):
    def __init__(self, img_size, channels):
        self.img_size= img_size
        self.channels = channels

    def build(self, hp):
        model = Sequential([
                    Input(shape=(self.img_size, self.img_size, self.channels)),
                    Flatten(),
                    
                    Dense(self.img_size, activation='relu'),
                    Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.5)),
                    
                    Dense(units=hp.Int('units', min_value=32, max_value=self.img_size*2, step=32, default=256), activation='relu'),
                    Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1, default=0.5)),
                    
                    Dense(1, 'sigmoid')
                ], name='mlp')
    
        model.compile(
                    loss='binary_crossentropy',
                    optimizer=Adam(
                            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
                    ),
                    metrics=['accuracy']
        )
        
        return model
    
    def name(self):
        return self.build(kt.HyperParameters()).name
    
    def param(self):
        hp=kt.HyperParameters()
        self.build(hp)
        return hp    