import tensorflow
from tensorflow import keras
import numpy as np

#laod the Fashion MNIST data
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

# Save classnames for later use
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Deviding pixel values by 255 to get them between 0-1 for easier training
train_images = train_images/255.0
test_images = test_images/255.0

#define a 2-layer neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

# compile the model defined above
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#train the model on the training data and save results in history object
history = model.fit(train_images,train_labels, epochs = 10, batch_size = 128)

#evaluate the model on the test dataset
test_loss,test_accuracy  = model.evaluate(test_images,test_labels)

print ("Test accuracy :",test_accuracy)
