from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten
from keras.optimizers import RMSprop


def get(input_shape=(28, 28, 1), kernel_size=(2, 2)):
    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=kernel_size,
            padding='Same',
            activation='relu',
            input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            filters=32,
            kernel_size=kernel_size,
            padding='Same',
            activation='relu'))
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            32, kernel_size=5, strides=2, padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(
        Conv2D(
            filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            64, kernel_size=5, strides=2, padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(10, activation="softmax"))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
        metrics=['accuracy'])

    return model