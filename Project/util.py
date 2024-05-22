import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from plot_keras_history import plot_history
import keras_tuner as kt


def import_images(dir_name,img_size=128, color_mode='rgb', batch_size=32):
    '''
    This function import all the images and create a list of this. 

    dir_name: is 
    img_size: set on 128
    color_mode: set on rgb or grayscale

    '''
    train_images = []
    train_labels = []

    train_ds = image_dataset_from_directory(
                    dir_name,
                    seed=123,
                    image_size=(img_size, img_size),
                    color_mode=color_mode,
                    shuffle=True,
                    batch_size=batch_size
                  )
    class_names = train_ds.class_names
    print('The classes name are:' + str(class_names))
    print('Number of training batches:', tf.data.experimental.cardinality(train_ds).numpy())

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    for images, labels in tqdm(train_ds):
        train_images.extend(images.numpy())
        train_labels.extend(labels.numpy())

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Print the shape of train_images and train_labels
    print("Shape of train_images:", train_images.shape)
    print("Shape of train_labels:", train_labels.shape)

    return train_images, train_labels

    
def save(data, dir_model, name_file, extension='txt'):
    path= dir_model + '/'+ name_file + '.' + extension
    f = open(path,'w')
    f.write(str(data))
    f.close()

    
def training(model, train_images, train_labels, trainDataGen, testDataGen, model_name=None, tuning = False, epochs=30, batch_size=30):
    if model_name is None:
        model_name=model.name()

    print('You choose ' + model_name.upper() +' to train ' + (str('with tuning') if tuning else str('without tuning')) + '.')
    

    model_name=model_name + str('_tuning') if tuning else model_name+str('_no_tuning')
    
    tuner = kt.RandomSearch(
                    model,
                    hyperparameters=model.param(),
                    objective='val_accuracy',
                    tune_new_entries=tuning,
                    max_trials=5,
                    directory='my_dir',
                    project_name = model_name
                )

    kfold = KFold(n_splits=5, shuffle=True)

    histories = []
    metrics = []
    fold_num = 1

    for train_index, val_index in kfold.split(train_images):

        tf.keras.backend.clear_session()

        print('Training Fold ' + str(fold_num) + '...')
        stop_early=EarlyStopping(monitor='val_loss',patience=5, verbose=1)

        model_path_name = 'models/'+ str(model_name) + 'best_model_Fold_' + str(fold_num) + '.keras'

        # import the images and apply Data Augmentation
        X_train = train_images[train_index].astype('float32')
        X_val = train_images[val_index].astype('float32')
        y_train = train_labels[train_index]
        y_val = train_labels[val_index]

        X_train, subval_X, y_train, subval_y = train_test_split(X_train, y_train, test_size=0.25)

        train_batch = trainDataGen.flow(
                            X_train,
                            y_train,
                            batch_size=batch_size
                          )
        test_batch =  testDataGen.flow(
                        X_val,
                        y_val,
                        batch_size=batch_size
                    )
        val_batch =  testDataGen.flow(
                        subval_X,
                        subval_y,
                        batch_size=batch_size
                    )
        #Define the model which will train
        if tuning:
            tuner.search(train_batch, epochs=epochs, validation_data=val_batch, callbacks=[EarlyStopping(monitor='val_loss',patience=5, verbose=1)])
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print('Hyper parameters: ' + str(best_hps.values))

            model=tuner.hypermodel.build(best_hps)
            if fold_num==1:
                print(model.summary())
        else:
            print('Hyper parameters: ' + str(model.param().values))
            model = tuner.hypermodel.build(model.param())
            
            if fold_num==1:
                print(model.summary())
        #try:
         #   model = tf.keras.models.load_model(model_path_name)
        #    print('Loaded successfully')
        #except(OSError, IOError):

        #training
        history = model.fit(
                          train_batch,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=test_batch,
                          verbose=1,
                          callbacks=[EarlyStopping(monitor='val_loss',patience=5, verbose=1),
                                     ModelCheckpoint(filepath=model_path_name, monitor='val_loss', save_best_only=True, mode='min')
                                    ]
                        )
        
        histories.append(history.history)
        plot_history(history)
        plt.savefig(str(model_name) + '/Fold_' + str(fold_num) + '.png')
        #model.save(model_name)

        #Evaluate the model 
        val_loss, val_accuracy = model.evaluate(test_batch,verbose=1)  
        metrics.append({'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                        'zero_one_loss': 1-val_accuracy})

        
        fold_num = fold_num + 1
        print('='*100)

    #Do the average of the metrics and save some data    
    average_metrics = {
        'avg_val_loss': sum(m['val_loss'] for m in metrics)/len(metrics),
        'avg_val_accuracy': sum(m['val_accuracy'] for m in metrics)/len(metrics),
        'avg_zero_one': sum(m['zero_one_loss'] for m in metrics)/len(metrics)
    }

    print('Average performance metrics:\n' + str(average_metrics))
    save(histories, model_name, 'histories')
    save(average_metrics, model_name,'metrics')