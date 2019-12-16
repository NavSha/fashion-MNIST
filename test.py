import keras
from keras.models import load_model

#load the model
model = load_model('fashion_mnist_weights.h5')
# summary of the model
model.summary()

#laod the Fashion MNIST data
fashion_mnist = keras.datasets.fashion_mnist
(_,_),(test_images,test_labels) = fashion_mnist.load_data()

# evaluate model on the test dataset 
loss, accuracy = model.evaluate(test_images,test_labels)

print("Accuracy of the model on test dataset: ", accuracy)
