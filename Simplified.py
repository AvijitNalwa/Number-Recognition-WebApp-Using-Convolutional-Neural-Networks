import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import datetime

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255 # normalize training data
X_test = X_test / 255 # normalize test data

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

def convolutional_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=20, kernel_size=(2, 2), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(28, 28, 1)))
    model.add(layers.SpatialDropout2D(0.3))
    #model.add(layers.BatchNormalization())

    # model.add(layers.Conv2D(filters=60, kernel_size=(2, 2), strides=(1, 1), activation='relu',
    #                         kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.Conv2D(filters=40, kernel_size=(2, 2), strides=(1, 1), activation='relu',
    #                         kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.Conv2D(filters=40, kernel_size=(2, 2), strides=(1, 1), activation='relu',
    #                         kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.SpatialDropout2D(0.1))
    # model.add(layers.Conv2D(filters=30, kernel_size=(2, 2), strides=(1, 1), activation='relu',
    #                         kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.SpatialDropout2D(0.1))
    model.add(layers.Conv2D(filters=20, kernel_size=(2, 2), strides=(1, 1), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Conv2D(filters=20, kernel_size=(2, 2), strides=(1, 1), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.SpatialDropout2D(0.4))

    model.add(layers.Conv2D(filters=30, kernel_size=(2, 2), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.SpatialDropout2D(0.3))
    #model.add(layers.SpatialDropout2D(0.1))
    #model.add(layers.BatchNormalization())

   # model.add(layers.Conv2D(filters=12, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    #model.add(layers.SpatialDropout2D(0.2))
    #model.add(layers.BatchNormalization())

    #model.add(layers.Conv2D(filters=10, kernel_size=(4, 4), strides=(1, 1), activation='relu',
                            #kernel_regularizer=regularizers.l1(0.001)))
    #model.add(layers.SpatialDropout2D(0.2))
    #model.add(layers.BatchNormalization())

    #model.add(layers.Conv2D(filters=10, kernel_size=(7, 7), strides=(1, 1), activation='relu',
                            #kernel_regularizer=regularizers.l1(0.01)))
    #model.add(layers.SpatialDropout2D(0.2))
    #model.add(layers.BatchNormalization())


    model.add(layers.Flatten())
    # model.add(layers.Dense(1000, activation='relu'))
    # model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(20, activation='relu'))
    # model.add(layers.Dense(50, activation='relu'))
    # model.add(layers.Dense(10, activation='relu'))
    # model.add(layers.Dense(10, activation='relu'))
    # model.add(layers.Dropout(0.3))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = convolutional_model()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.20, # Randomly zoom image
        width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)
        #shear_range=8.0)  # randomly flip images

#datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=50),
                              epochs=20, validation_data=(X_test,y_test),
                              verbose=1, callbacks=[tensorboard_callback])

# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=60, verbose=1, callbacks=[tensorboard_callback, es])

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1]*100, 100-scores[1]*100))
model.save('mnist_4-12_13(aug).h5')