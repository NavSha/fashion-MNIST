import keras
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras import models,layers
from keras.applications import VGG16
from keras.optimizers import Adam

#load the Fashion MNIST data
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

# summarize loaded dataset
print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))
# plot first few images
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

# Dividing pixel values by 255 to get them between 0-1 for easier training
train_images = train_images.reshape((60000,28,28,1))
train_images = train_images/255.0
test_images = test_images.reshape((10000,28,28,1))
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

#define a small CNN network
def create_model():
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
  return model

model = create_model()

# print summary of the model architecture
model.summary()

# compile the model defined above
opt = Adam(lr=0.001)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Add callbacks for more efficient training of the models
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
stop = EarlyStopping(monitor = 'val_loss', patience = 10,verbose=1)
checkpoint = ModelCheckpoint(filepath='fashion_mnist_weights.h5',monitor='val_loss',verbose = 1, save_best_only = True)
reduce = ReduceLROnPlateau(monitor='val_loss',factor = 0.1, patience = 3, verbose = 1)
#tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, batch_size=128, write_graph = True, write_grads = True, write_images = True)

#train the model on the partial training data and save results in history object
history = model.fit(partial_train_images,partial_train_labels, epochs = 100, batch_size = 128, validation_data = (val_train_images,val_train_labels), callbacks = [stop,checkpoint,reduce])

#save model weights in a h5 file
model.save('fashion_mnist_weights.h5')
#save model in a json file
model_json = model.to_json()
with open('fashion_mnist_model.json',"w") as json_file:
    json_file.write(model_json)

#plot loss and validation curves during Training
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
