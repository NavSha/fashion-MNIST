import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import models,layers
from keras.applications import VGG16
from keras.optimizers import Adam
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Save class names for later use
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_dataset():
     ''' Loads and reshapes the in-built Fashion MNIST dataset from Keras.

     Returns:
     train_images, train_labels: training images and labels
     '''

     fashion_mnist = keras.datasets.fashion_mnist
     (train_images,train_labels),(_,_) = fashion_mnist.load_data()
     # Reshape data by adding a channel axis
     train_images = train_images.reshape((60000,28,28,1))
     return train_images, train_labels

def normalize_dataset(trainX):
    ''' Normalizes the training images to be used for model training.

    Args:
        trainX(ndarray): Training images

    Returns:
        train_norm(ndarray): Normalized training images
    '''

    train_norm = trainX/255.0
    return train_norm

def create_model():
    ''' Create the model using ConvNet layers.

    Returns:
        model(Sequential): A compiled model as per the specified design
    '''

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
    model.add(layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu',kernel_initializer='he_uniform'))
    model.add(layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu',kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(10, activation = 'softmax'))

    # compile the model
    optimizer = Adam(lr=0.001)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model

def prepare_val_data(trainX, trainY):
    ''' Prepares data for model validation by keeping 10000 samples for validation and rest for training.

    Args:
        trainX (ndarray): Training images
        trainY (ndarray): Training labels

    Returns:
        partial_trainX, valX(ndarray): partial training images and validation images
        partial_trainY, valY(ndarray): partial training labels and validation labels
    '''

    partial_trainX = trainX[10000:]
    partial_trainY = trainY[10000:]
    valX  = trainX[:10000]
    valY = trainY[:10000]

    return partial_trainX, partial_trainY, valX, valY


def train_and_compile_model(trainX, trainY, valX, valY):

    ''' Trains and compiles the model using the partial training data and validation data.

    Args:
        trainX (ndarray): Training images
        trainY (ndarray): Training labels
        valX (ndarray): Validation images
        valY(ndarray): Validation labels

    Returns:
        history(list): A list of loss and accuracy of the model over epochs
    '''

    # create model
    model = create_model()

    #Add callbacks for more efficient training of the models
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stop = EarlyStopping(monitor = 'val_loss',
                        patience = 10,verbose=1)
    checkpoints = ModelCheckpoint(filepath='fashion_mnist_weights.h5',
                                monitor='val_loss',verbose = 1,
                                save_best_only = True)
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                            factor = 0.1,
                                            patience = 3,
                                            verbose = 1)
    tensorboard = TensorBoard(log_dir=log_dir,
                            histogram_freq=1,
                            batch_size=128,
                            write_graph = True,
                            write_grads = True,
                            write_images = True)

    #train the model on the partial training data and save results in history object
    history = model.fit(trainX, trainY,
                        epochs = 100,
                        batch_size = 128,
                        validation_data = (valX,valY),
                        callbacks = [stop,checkpoints,reduce_learning_rate,tensorboard])

    return history

def save_model():
    ''' Saves the model as .h5 and json files for later use.'''

    model.save('fashion_mnist_weights.h5')
    model_json = model.to_json()
    with open('fashion_mnist_model.json',"w") as json_file:
        json_file.write(model_json)

def plot_loss_and_accuracy(history):
    ''' Plots loss and validation curves during training.

    Args:
        history (list): List that contains loss and accuracy numbers for each epoch
    '''

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc)+1)

    plt.plot(epochs,loss,'ro',label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

def run_validation_harness():
    ''' It calls the functions defined above sequentially and plots model performance during training.'''

    # load dataset
    train_images, train_labels = load_dataset()
    # prepare normalized data
    train_images = normalize_dataset(train_images)
    #create model
    model = create_model()
    #prepare validation data
    partial_train_images, partial_train_labels, val_images, val_labels = prepare_val_data(train_images, train_labels)
    # train the model
    history = train_and_compile_model(partial_train_images, partial_train_labels, val_images, val_labels)
    #save model details
    save_model()
    # learning curves
    plot_loss_and_accuracy(history)

if __name__ == "__main__":
    run_validation_harness()
