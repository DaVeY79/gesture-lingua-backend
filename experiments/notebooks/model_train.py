
# coding: utf-8
import sys
sys.path.append("/Users/davidabraham/gesture-lingua-backend/")

# Import packages and set numpy random seed
import numpy as np
np.random.seed(5)
import tensorflow as tf
tf.set_random_seed(2)
from datasets import sign_language
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
import logging

### Enabling logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def split_data():
    (x_train, y_train), (x_test, y_test) = sign_language.load_data() # Load pre-shuffled training and test datasets
    labels = ['A','B','C'] # Store labels of dataset
    y_train_OH = to_categorical(y_train) # One-hot encode the training labels
    y_test_OH = to_categorical(y_test) # One-hot encode the test labels
    return x_train, y_train_OH, x_test, y_test_OH

def main():
    """
    Definining a convolutional neural network to classify the data.This network accepts an image of an
    American Sign Language letter as input. The output layer returns the network's predicted probabilities
    that the image belongs in each category.
    """
    # create a file handler
    handler = logging.FileHandler('Log_Model_Builder.log')
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    logger.info("Loading image data")

    x_train, y_train, x_test, y_test = split_data()

    model = Sequential()
    # First convolutional layer accepts image input
    model.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu',
                    input_shape=(50, 50, 3)))
    # Add a max pooling layer
    model.add(MaxPooling2D(pool_size =(4,4)))
    # Add a convolutional layer
    model.add(Conv2D(filters = 15,kernel_size=5,padding = 'same',activation='relu'))
    # Add another max pooling layer
    model.add(MaxPooling2D(pool_size =(4,4)))
    # Flatten and feed to output layer
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    # Summarize the model
    #model.summary()

    # Compile the model
    logger.debug(model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']))

    # Train the model
    hist = model.fit(x_train,y_train,batch_size=32,validation_split=0.2,epochs=2)

    """
    Test the model
    To evaluate the model, we'll use the test dataset.  This will tell us how the network performs when
    classifying images it has never seen before!If the classification accuracy on the test dataset is similar
    to the training dataset, this is a good sign that the model did not overfit to the training data.
    """

    # Obtain accuracy on test set
    score = model.evaluate(x=x_test,y=y_test, verbose=0)
    logger.debug('Test accuracy: %f', score[1])

    model.save("model.h5")     #save model

    y_probs = model.predict(x_test) # Get predicted probabilities for test dataset

    # Get predicted labels for test dataset
    y_preds = np.argmax(y_probs,axis = 1)
    logger.info("Model creation completed")
    #print("Predictions are : {}".format(y_preds))

if __name__ == "__main__":
    main()
