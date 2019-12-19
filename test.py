import keras
from keras.models import load_model

def test_model():
    ''' Tests a model using the test dataset '''
    #load the model
    model = load_model('fashion_mnist_weights.h5')
    # summary of the model
    model.summary()

    #load the Fashion MNIST data
    fashion_mnist = keras.datasets.fashion_mnist
    (_,_),(test_images,test_labels) = fashion_mnist.load_data()
    # reshape and normalize test data
    test_images = test_images.reshape((10000,28,28,1))
    test_images = test_images/255.0

    # evaluate model on the test dataset
    loss, accuracy = model.evaluate(test_images,test_labels)

    print("Accuracy of the model on test dataset: ", accuracy)

if __name__ == "__main__":
    test_model()
