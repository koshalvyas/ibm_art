#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install adversarial-robustness-toolbox')


# In[8]:


import keras
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
from keras.utils.np_utils import to_categorical

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist
from tensorflow.keras.datasets import cifar10
from keras.optimizers import SGD
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout

import tensorflow_hub as hub

tf.compat.v1.disable_eager_execution()

#url = 'https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4'
#base_model = hub.KerasLayer(url, input_shape=(32, 32, 3), trainable=True)


# ## Data Loading

# In[4]:


# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype("float32") 
#x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype("float32")

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

min_pixel_value = x_train.min()
max_pixel_value = x_train.max()


# ##Conv2d Model

# In[5]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
# compile model
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# ART Classifier

# In[6]:


# Step 3: Create the ART classifier

classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

# Step 4: Train the ART classifier

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=100)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on clean test examples: {}%".format(accuracy * 100))


# FGSM Network

# In[7]:


# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


# 
# ##Clean and Pertubated Samples

# In[21]:


for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(x_train[i])
# show the figure
pyplot.show()


# In[8]:


from matplotlib import pyplot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(x_test_adv[i])
# show the figure
pyplot.show()


# In[9]:


from scipy.io import savemat
test_adv = {"pertubated samples": x_test_adv, "label": y_test}
savemat("test_adversaries_FGSM.mat", test_adv)


# In[10]:


from skimage.measure import compare_ssim as ssim
from statistics import mean 
s = []
for i in range(0,len(x_test)):
  
  s.append(ssim(x_test[i],x_test_adv[i], multichannel=True))

print(mean(s))  


# ##Adversary Detection: BinaryInputDetector

# In[11]:


from art import config
from keras.models import load_model
from art.utils import load_dataset, get_file
from art.defences.detector.evasion import BinaryInputDetector


# In[12]:


path = get_file('BID_eps=0.05.h5',extract=False, path=config.ART_DATA_PATH,
                url='https://www.dropbox.com/s/cbyfk65497wwbtn/BID_eps%3D0.05.h5?dl=1')
detector_model = load_model(path)
detector_classifier = KerasClassifier(clip_values=(-0.5, 0.5), model=detector_model, use_logits=False)
detector = BinaryInputDetector(detector_classifier)


# In[13]:


detector_model.summary()


# In[14]:


x_train_adv = attack.generate(x_train)
nb_train = x_train.shape[0]

x_train_detector = np.concatenate((x_train, x_train_adv), axis=0)
y_train_detector = np.concatenate((np.array([[1,0]]*nb_train), np.array([[0,1]]*nb_train)), axis=0)


# In[15]:


detector.fit(x_train_detector, y_train_detector, nb_epochs=2, batch_size=200)


# In[16]:


flag_adv = np.sum(np.argmax(detector.predict(x_test_adv), axis=1) == 1)

print("Adversarial test data (first 100 images):")
print("Flagged: {}".format(flag_adv))
print("Not flagged: {}".format(10000 - flag_adv))


# In[17]:


flag_original = np.sum(np.argmax(detector.predict(x_test[:100]), axis=1) == 1)

print("Original test data (first 100 images):")
print("Flagged: {}".format(flag_original))
print("Not flagged: {}".format(100 - flag_original))


# ##Adversary Mitigation: Filtering

# In[18]:


from art.defences.transformer.poisoning import NeuralCleanse
#from art.estimators.certification.neural_cleanse import KerasNeuralCleanse


# In[19]:


from art.estimators.poison_mitigation.neural_cleanse import KerasNeuralCleanse


# In[20]:


cleanse = NeuralCleanse(classifier)
defence_cleanse = cleanse(classifier, steps=10, learning_rate=0.1)


# In[ ]:


pattern, mask = defence_cleanse.generate_backdoor(x_test, y_test, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
plt.imshow(np.squeeze(mask * pattern))


# In[ ]:


defence_cleanse = cleanse(classifier, steps=10, learning_rate=0.1)
defence_cleanse.mitigate(clean_x_test, clean_y_test, mitigation_types=["filtering"])


# In[ ]:


poison_pred = defence_cleanse.predict(poison_x_test)
num_filtered = np.sum(np.all(poison_pred == np.zeros(10), axis=1))
num_poison = len(poison_pred)
effectiveness = float(num_filtered) / num_poison * 100
print("Filtered {}/{} poison samples ({:.2f}% effective)".format(num_filtered, num_poison, effectiveness))


# In[ ]:




