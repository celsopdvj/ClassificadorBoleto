#from _future_ import absolute_import, division, print_function

# tensorflow e keras
import tensorflow as tf
from tensorflow import keras

# helpers
import numpy as np
import matplotlib.pyplot as plt

# imagem
from PIL import Image
import scipy.ndimage
import glob

def findImages(dir):
    images = []
    for filename in glob.glob(dir):
        im = scipy.ndimage.imread(filename,True)
        images.append(np.array(im))
    return np.array(images)

print('Caminho da imagem para predição: ')
pathImageToPredict = input()

# classes
class_names = ['Boleto','Nao Boleto']

# imagens de treinamento de boleto
train_images = findImages('treino/boleto/*.*')
train_labels = [0] * train_images.shape[0] 

# imagens de treinamento de nao boleto
img_temp = findImages('treino/naoboleto/*.*')
train_images = np.append(train_images, img_temp, axis=0)
train_labels = np.append(train_labels, [1] * img_temp.shape[0], axis=0)

train_labels = np.array([0,0,0,1,1,1,1])

# imagens de teste de boleto
test_images = findImages('test/boleto/*.*')
test_labels = [0] * test_images.shape[0] 

# imagens de teste de nao boleto
img_temp = findImages('test/naoboleto/*.*')
test_images = np.append(test_images, img_temp, axis=0)
test_labels = np.append(test_labels, [1] * img_temp.shape[0], axis=0)

# Layer - Extrai representacao dos dados para a layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(600, 800)),
    keras.layers.Dense(28, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Compila o model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Associando imagens de treino a model
model.fit(train_images, train_labels, epochs=5)

# Avaliar a acuracia
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Acuracia do teste:', test_acc)
print('\n\n')

imageToPredict = findImages(pathImageToPredict)
predictions_single = model.predict(imageToPredict)
predict = np.argmax(predictions_single)
print('Predição da sua image:', class_names[predict])