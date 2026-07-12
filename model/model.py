import keras
from keras import layers

"""
Начальные условия: 128 токенов по 32 сжатые амплитуды
"""

TOKEN_COUNT = 128

TOKEN_LENGTH = 32


"""
На входном слое находится взаимосвязь каждых <INPUT_KERNEL> токенов с шагом <INPUT_DIL> 

SQUEEZE_KERNEL сделан для сжатия и выделения самых важных частей из <FILTERS_2> параметров (условных уже преобразованных амплитуд)

Первые слои активируются функцией <START_ACTIV>
"""

INPUT_KERNEL = 16

SQUEEZE_KERNEL = 1

FILTERS_1 = 64

FILTERS_2 = 16

INPUT_DIL = 2

START_ACTIV = "tanh"

DICT_SIZE = 1000 # Условная тысяча уникальных токенов

model = keras.models.Sequential()

model.add(layers.Conv1D(filters=FILTERS_1, kernel_size=INPUT_KERNEL, dilation_rate=INPUT_DIL, activation=START_ACTIV, input_shape=(TOKEN_COUNT, TOKEN_LENGTH)))

model.add(layers.Conv1D(filters=FILTERS_2, kernel_size=SQUEEZE_KERNEL, activation=START_ACTIV))

model.add(layers.Flatten())

model.add(layers.Dense(DICT_SIZE, activation="softmax"))

optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)