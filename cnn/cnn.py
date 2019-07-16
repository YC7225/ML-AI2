import mnist
import numpy as np
import cv2 
#from PIL impsort Image
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
import os
from PIL import Image
path1 = 'C:/Users/hp/OneDrive/Desktop/img of MLAI Notes'
##path2 = 'C:/Users/hp/OneDrive/Desktop/rr'
listing = os.listdir(path1)
mmatrix = np.array([np.array(Image.open(path1 + '\\' + im2)).flatten()
              for im2 in listing],'f')

train_images = mmatrix
train_labels = [0,1,2,3,4,5,6,7,8,9]
'''
train_images[0] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_8.png',0)
train_labels[0] = 8
# Normalize the images.
#train_images = train_images[:1000]
#train_labels = train_labels[:1000]
#test_images = mnist.test_images()[9999:]
#test_labels = mnist.test_labels()[9999:]
'''
test_images = cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_1three.png',0)
test_images = test_images.reshape(1,28,28)
test_labels = [3]
# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
#train_images = mnist.train_images()[:1001]
#train_labels = mnist.train_labels()[:1001]
#train_images[1000] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_1rsz_pc7r5dlli.png',0)
'''
train_images[10002] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_one.png',0)
train_images[10003] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_two.png',0)
train_images[10004] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_1three.png',0)
train_images[10005] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_four.png',0)
train_images[10006] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_five.png',0)
train_images[10007] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_six.png',0)
train_images[10008] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_seven.png',0)
train_images[10009] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_8.png',0)
train_images[10010] =  cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_nine.png',0)
'''
#train_labels[1000] = 1
'''
train_images[10002] = 1
train_images[10003] = 2
train_images[10004] = 3
train_images[10005] = 4
train_images[10006] = 5
train_images[10007] = 6
train_images[10008] = 7
train_images[10009] = 8
train_images[10010] = 9
'''

#train_images = train_images[:1000]
#train_labels = train_labels[:1000]
#test_images = mnist.test_images()[9999:]
#test_labels = mnist.test_labels()[9999:]
#test_images = cv2.imread('C:/Users/hp/OneDrive/Desktop/img of MLAI Notes/rsz_1rsz_pc7r5dlli.png',0)
#test_images = test_images.reshape(1,28,28)
#test_labels = [1]
#print(test_images.shape)a 
#a = train_images[0]
#im = Image.fromarray(a)
#b = im.show('image', test_images)
#b
conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
  # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with. This is standard practice.
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)
  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

def train(im, label, lr=.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc

print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(2):
  print('--- Epoch %d ---' % (epoch + 1))   

  # Shuffle the training data
  #permutation = np.random.permutation(len(train_images))
  #train_images = train_images[permutation]
  #train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i > 0 and i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)