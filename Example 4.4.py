# Example 4.4 LeNet-5 Keras (Diperbaiki)
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D
from keras import optimizers

model = Sequential()

# Menambahkan layer Input terpisah untuk mengatasi UserWarning
model.add(Input(shape=(32, 32, 1)))

# Layer Conv2D pertama tanpa argumen input_shape
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))

# Menambahkan argumen pool_size untuk mengatasi TypeError
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

# Menambahkan argumen pool_size untuk mengatasi TypeError
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()