import tensorflow
from tensorflow import keras
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#laod the Fashion MNIST data
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

# Deviding pixel values by 255 to get them between 0-1 for easier training
train_images = train_images/255.0
test_images = test_images/255.0

#lets create validation data from the training dataset
val_train_images  = train_images[:10000]
val_train_labels = train_labels[:10000]

#Remaining training data
partial_train_images = train_images[10000:]
partial_train_labels = train_labels[10000:]

# Save classnames for later use
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#define a 2-layer neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

# compile the model defined above
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#train the model on the partial training data, find training and validation acc
#and loss and save results in history object
history = model.fit(train_images,train_labels, epochs = 5, batch_size = 128, validation_data = (val_train_images,val_train_labels))

#save model weights in a h5 file
model.save('fashion_mnist_weights.h5')
#save model in a json file
model_json = model.to_json()
with open('fashion_mnist_model.json',"w") as json_file:
    json_file.write(model_json)

#plot loss and validation curves during Training
import matplotlib.pyplot as plt
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

#evaluate the model on the test dataset
test_loss,test_accuracy  = model.evaluate(test_images,test_labels)

print ("Test accuracy :",test_accuracy)
