import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
import keras


# Set the memory growth to true for the GPU to expand the memory as needed
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')


# Load the data

# Read data from a csv file using pandas

train_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/train.csv')

# Shuffle the data

train_data = shuffle(train_data)

# Take the images and labels
# Images are on the first column and labels are on the second column

images = train_data.iloc[:,0]
labels = train_data.iloc[:,1]

train_images = []
train_labels = []

for image in images: # for each image in the images list load the image, resize it to 64x64 and append it to the train_images list
    img = cv2.imread('/kaggle/input/unibuc-dhc-2023/train_images/'+image)
    img = cv2.resize(img, (64, 64))
    img = np.array(img)
    img = img/255.0 # normalize the image
    train_images.append(img)
    
train_images = np.array(train_images)
    
for label in labels: # for each label in the labels list append it to the train_labels list
    train_labels.append(label)
    
train_labels = np.array(train_labels)
    
plt.figure(figsize=(10,10)) # plot the first 9 images with their labels to see if the data is loaded correctly
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.xlabel(train_labels[i])
    
# Give them a good shuffle to make sure that the model doesn't catch any patterns given by the order of the images
train_images, train_labels = shuffle(train_images, train_labels, random_state=14)

# data augmentation

from keras.preprocessing.image import ImageDataGenerator

# image generator from keras that creates new images from the existing ones by applying different transformations
datagen = ImageDataGenerator( 
    rotation_range=10,
    vertical_flip=False,
    horizontal_flip=True,
    zoom_range=-0.5
)

datagen.fit(train_images) # fit the generator on the train images (actual transformation is done her)

augmented_images, augmented_labels = next(datagen.flow(train_images, train_labels, batch_size=len(train_images))) # Collect the new images and labels

# concatenate the original images with the augmented images
train_images = np.concatenate((train_images, augmented_images))
train_labels = np.concatenate((train_labels, augmented_labels))

# shuffle the data again

train_images, train_labels = shuffle(train_images, train_labels, random_state=14)

# Read validation data with the same principle as the train data

val_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/val.csv')

images = val_data.iloc[:,0]
labels = val_data.iloc[:,1]

val_images = []
val_labels = []

for image in images:
    img = cv2.imread('/kaggle/input/unibuc-dhc-2023/val_images/'+image)
    img = cv2.resize(img, (64,64))
    img = np.array(img)
    img = img/255.0
    val_images.append(img)

for label in labels:
    val_labels.append(label)

val_images = np.array(val_images)
val_labels = np.array(val_labels)

# Plot the images

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(val_images[i], cmap='gray')
    plt.xlabel(val_labels[i])

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout, RandomRotation, RandomFlip, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import SeparableConv2D

# Convert the images and labels to tensors and create a dataset
# Prefetch and cache the dataset for better performance

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_images_tensor = tf.convert_to_tensor(train_images)
train_labels_tensor = tf.convert_to_tensor(train_labels)

val_images_tensor = tf.convert_to_tensor(val_images)
val_labels_tensor = tf.convert_to_tensor(val_labels)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32).cache().prefetch(AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32).cache().prefetch(AUTOTUNE)

################# 3. Build Model #################

model = Sequential()

# ARHITECTURA VGG16 CARE ESTE LUATA DIN CURSUL DE ML

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides = 1, activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2 ), strides=(2, 2)))
model.add(Dropout(0.6))

model.add(RandomFlip('horizontal_and_vertical'))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2 ), strides=(2, 2)))
model.add(Dropout(0.6))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2 ), strides=(2, 2)))
model.add(Dropout(0.6))

model.add(Conv2D(filters=512, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2 ), strides=(2, 2)))
model.add(Dropout(0.6))

model.add(Conv2D(filters=512, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), strides = 1, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2 ), strides=(2, 2)))
model.add(Dropout(0.6))

# augumentation layer
model.add(RandomRotation(3))

model.add(Flatten())

model.add(Dense(4069, activation='relu'))
model.add(Dense(4069, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.7))
model.add(Dense(96, activation='softmax'))
# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# print model summary
model.summary()

# print(train_images.shape, val_images.shape)

# model = None
# hist = None

import gc # garbage collector for cleaning unused memory and freeing up some space
gc.collect()

# lr scheduler

from keras.callbacks import ReduceLROnPlateau
# reduce learning rate when a metric has stopped improving (val_accuracy in this case) to make the model converge better and reduce overfitting
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=22, verbose=1, min_lr=0.0001, mode='auto', initial_lr=0.001)


# callbacks

from keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5" # save the best model based on validation accuracy

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False, restore_best_weights=True)

callbacks_list = [checkpoint, lr_scheduler] # list of callbacks namely the checkpoint and the lr_scheduler

# train model

history = model.fit(train_dataset, epochs=250, batch_size=32, validation_data=val_dataset, callbacks=callbacks_list)

# evaluate model
model.load_weights('weights.best.hdf5') # load the best model based on validation accuracy

model.evaluate(val_images, val_labels)

# confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(val_images)

cm = confusion_matrix(val_labels, y_pred.argmax(axis=1))

plt.figure(figsize=(96, 96))
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

test_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/test.csv')

images = test_data.iloc[:,0]

test_images = []
test_labels = []

for image in images:
    img = cv2.imread('/kaggle/input/unibuc-dhc-2023/test_images/'+image)
    img = cv2.resize(img, (64,64))
    img = np.array(img)
    img = img/255.0
    test_images.append(img)

test_images = np.array(test_images)

pred = model.predict(test_images)

csv_file = open('submission.csv', 'w')
csv_file.write('Image,Class\n')

for i in range(len(pred)):
    csv_file.write(test_data.values[i][0] + ',' + str(np.argmax(pred[i])) + '\n')

csv_file.close()