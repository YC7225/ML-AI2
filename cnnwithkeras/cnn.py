import numpy as np
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

#import os
#from PIL import Image
'''
path1 = 'C:/Users/hp/OneDrive/Desktop/img of MLAI Notes'
##path2 = 'C:/Users/hp/OneDrive/Desktop/rr'
listing = os.listdir(path1)
mmatrix = np.array([np.array(Image.open(path1 + '\\' + im2)).flatten()
              for im2 in listing],'f')

train_images = mmatrix
train_labels = [0,1,2,3,4,5,6,7,8,9]

train_images[0] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_8.png',0)
train_labels[0] = 8
# Normalize the images.
#train_images = train_images[:1000]x
#train_labels = train_labels[:1000]
#test_images = mnist.test_images()[9999:]
#test_labels = mnist.test_labels()[9999:]
'''
# batch_size = 128
num_classes = 10
epochs = 12

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images =  cv2.imread('C:/Users/hp/Downloads/testSample/testSample/img_28.jpg',0)
test_labels = [8]
test_labels = np.asarray(test_labels)
test_labels = to_categorical(test_labels,10)
#test_labels = test_labels.reshape(9,10)
train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(1,28,28,1)
#test_images = test_images.reshape(10000,28,28,1)
train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)
#test_labels = 8
model = Sequential([
  Conv2D(8, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)),
  Conv2D(16, (3, 3), activation='relu'),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(32, (3, 3), activation='relu'),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D(pool_size=(2, 2)),
  Dropout(0.75),
  Flatten(),
  #Dense(64, activation='sigmoid', input_shape=(784,)),
  #Dense(64, activation='sigmoid'),
  #Dense(64, activation='relu'),
  Dense(num_classes, activation='softmax'),
])


model.load_weights('model.h5')

model.compile(
  optimizer='Adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

''';'
model.fit(
  train_images,
  train_labels,
  epochs=10,
  verbose=1,
  # validation_data=(test_images, test_labels)
  batch_size=64
)

#model.save_weights('model.h5')
'''
result = model.evaluate(
        test_images,
        test_labels
        )


print('total loss: ',result[0])
print('accuracy: ',result[1])
