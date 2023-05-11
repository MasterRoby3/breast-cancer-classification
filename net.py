import numpy as np
from matplotlib import pyplot as plt
import pandas as pd # for reading and writing tables
import scipy

# Define Drive folder from which data can be downloaded
data_folder = "../deeplearning/training_set/"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)    

train_generator = datagen_train.flow_from_directory(
    data_folder,
    seed=42,
    target_size=(224, 224),
    batch_size=32, 
    shuffle=True,
    class_mode='binary',
    subset='training')

validation_generator = datagen_val.flow_from_directory(
    data_folder,
    seed=42,
    target_size=(224, 224),
    batch_size=32, 
    shuffle=True,
    class_mode='binary',
    subset='validation')


input_shape = (224, 224, 3)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import metrics


def classic():
    model = Sequential()
    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model.add(ResNet50(include_top = False, pooling = 'avg', weights='imagenet'))
    # This layers as Dense for 2-class classification, i.e., dog or cat
    # using SoftMax activation for last layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = False

    return model


model = classic()

print("\n\n---------------------------------------------------------\n\n")
model.summary()

# Compile the model
optimizer = Adam(learning_rate=0.00001)

model.compile(
  optimizer=optimizer,
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=[metrics.SparseCategoricalAccuracy()])

# Modules to improve training (early_stopping, checkpoint and learning rate reduction when no improvement is detected)
early_stopping = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 25, verbose = 1)
mc = ModelCheckpoint ('best_model.h5', monitor = 'val_loss', mode = 'min', save_best_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)

# weight for classes based on data
class_weight = {0: 0.707, 1: 0.293}

# model 1st step training
history = model.fit(train_generator, epochs = 20, verbose = 1, validation_data = validation_generator, callbacks = [early_stopping, mc, reduce_lr], batch_size=32, class_weight=class_weight)

# Switch ResNet to trainable, lower learning rate, 2nd training step
model.layers[0].trainable = True
K.set_value(model.optimizer.learning_rate, 0.00001)
history = model.fit(train_generator, epochs = 50, verbose = 1, validation_data = validation_generator, callbacks = [early_stopping, mc, reduce_lr], batch_size=32, class_weight=class_weight)

# Plot results
import matplotlib.pyplot as plt
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()


# Code for external testing and submissions
test_path = "../deeplearning/"

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    classes = ['testing_set']
)

filenames = test_generator.filenames
nb_samples = len(filenames)

tst_predictions = model.predict_generator(test_generator, steps = nb_samples, verbose=1)

tst_predictions = tst_predictions[:,1]
sample_submission = pd.read_csv("../submissions/sample_submission_auc.csv")
submission = pd.DataFrame()
submission['Id'] = sample_submission['Id']
submission['Predicted'] = tst_predictions[:]
submission.to_csv('../submissions/submission_dl.csv', index=False)
