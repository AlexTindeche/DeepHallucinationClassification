import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
import keras

# Load the data

# Read data from a csv file using pandas

train_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/train.csv')

# Shuffle the data to make sure that the model doesn't catch any patterns given by the order of the images

train_data = shuffle(train_data)

# Take the images and labels
# Images are on the first column and labels are on the second column

images = train_data.iloc[:,0] # take all the rows from the first column
labels = train_data.iloc[:,1] # take all the rows from the second column

train_images = []
train_labels = []

for image in images: # for each image in the images list load the image, resize it to 64x64 and append it to the train_images list
    img = cv2.imread('/kaggle/input/unibuc-dhc-2023/train_images/'+image)
    img = cv2.resize(img, (64, 64)) # resize the image to 64x64
    img = np.array(img)
    img = img/255.0 # normalize the image
    train_images.append(img)
    
train_images = np.array(train_images)
    
for label in labels:
    train_labels.append(label)
    
train_labels = np.array(train_labels)
    
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.xlabel(train_labels[i])
    
# Give them a good shuffle 
train_images, train_labels = shuffle(train_images, train_labels, random_state=14)

# Read validation data using the same method as above

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


# KNN

class KNN:
    def __init__ (self, k, train_images, train_labels): # initialize the class with the number of neighbours, the train images and the train labels
        self.k = k
        self.train_images = train_images
        self.train_labels = train_labels
    
    def d1 (self, test_img):
        return np.sum(np.abs(self.train_images - test_img), axis = 1) # Manhattan distance
    
    def d2 (self, test_img):
        return np.sqrt(np.sum(np.square(self.train_images - test_img), axis = 1)) # Euclidean distance
    
    def d3 (self, test_img):
        return np.sum(self.train_images * test_img, axis = 1) / (np.sqrt(np.sum(np.square(self.train_images), axis = 1)) * np.sqrt(np.sum(np.square(test_img)))) # Cosine similarity

    def classify_image(self, test_img, distance_type = 1): # classify the image based on the distance type
        if distance_type > 3 or distance_type < 1: # if the distance type is not 1, 2 or 3 return an error
            print('Wrong distance type!')
            return
        if distance_type == 1: # if the distance type is 1 use the Manhattan distance
            distances = self.d1(test_img)
        elif distance_type == 2: # if the distance type is 2 use the Euclidean distance
            distances = self.d2(test_img)
        else: # if the distance type is 3 use the Cosine similarity
            distances = self.d3(test_img)

        distances = np.array(distances) # convert the distances to a numpy array
        sorted_distances = np.argsort(distances) # sort the distances and return the indices of the sorted array
        k_nearest_labels = self.train_labels[sorted_distances[:self.k]] # take the first k labels from the sorted array

        return np.argmax(np.bincount(k_nearest_labels)) # return the most common label from the k nearest labels

    def predict(self, test_images, distance_type = 1): # predict the labels for the test images
        predictions = [] # initialize the predictions list
        for test_img in test_images: # for each test image classify it and append the result to the predictions list
            predictions.append(self.classify_image(test_img, distance_type)) 
            # Print the progress
            print("Images left to predict: ", len(test_images) - len(predictions)) 

        return predictions


import time

# Train the model

neighbours = [3, 5, 7, 9, 11]
accuracy = []

for distance_type in range(1, 4): # for each distance type train the model and print the accuracy
    for k in neighbours: # for each number of neighbours train the model and print the accuracy
        knn = KNN(k, train_images.reshape(-1, 64*64*3), train_labels) # reshape the train images to a 2D array

        # Start timer
        start = time.time()

        predictions = knn.predict(val_images.reshape(-1, 64*64*3), distance_type) 

        # Stop timer
        end = time.time()

        print('Time elapsed: ', end - start)

        correct = np.count_nonzero(predictions == val_labels)
        print('Accuracy for k = ' + str(k) + ' is ' + str(correct/len(val_labels)))
        accuracy.append(correct/len(val_labels))
        
        # Confusion matrix 

        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(val_labels, predictions)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(96,96))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix for distance type ' + str(distance_type))
        plt.show()

    plt.plot(neighbours, accuracy)
    plt.xlabel('Number of neighbours')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for distance type ' + str(distance_type))
    plt.show()