# Example 4.9
# Simple CNN for the MNIST Dataset
import matplotlib.pyplot as plt # Tambahan import untuk plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten # Tambahan Input
from keras.layers import Conv2D, MaxPooling2D # Penulisan import yang lebih standar
from keras.utils import to_categorical # Perbaikan import to_categorical

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as gray scale
plt.subplot(141)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show()

# reshape to be [samples][width][height][channels] dan digabung dalam 1 baris
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs (menggunakan to_categorical langsung)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create model
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# UBAH INI JIKA INGIN SAMA PERSIS DENGAN BUKU (Meskipun sebenarnya salah secara logika untuk MNIST)
model.add(Dense(2, activation='softmax')) 

# TAMBAHKAN BARIS INI UNTUK MENAMPILKAN TABEL OUTPUT
model.summary()