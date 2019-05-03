import numpy as np
from src import ModelUtil, Model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.utils import plot_model
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# 训练参数
batch_size = 86
nb_epoch = 40
nb_classes = 10

# 图片尺寸
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# 固定随机种子
seed = 7
np.random.seed(seed)

#格式化数据

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 分配验证集
random_state = 7
X_train2, X_val2, Y_train2, Y_val2 = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=random_state)

# 获取模型
model = Model.get(input_shape)

# 学习速度递减
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.000005)

# TensorBoard文件生成
tbCallBack = TensorBoard(
    log_dir='./logs',  # log 目录
    histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #                  batch_size=32,     # 用多大量的数据计算直方图
    write_graph=True,  # 是否存储网络结构图
    write_grads=True,  # 是否可视化梯度直方图
    write_images=True,  # 是否可视化参数
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None)

# 数据增强设置
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1)

# 训练
history = model.fit_generator(
    datagen.flow(X_train2, Y_train2, batch_size=batch_size),
    epochs=nb_epoch,
    validation_data=(X_val2, Y_val2),
    verbose=1,
    steps_per_epoch=X_train.shape[0],
    callbacks=[learning_rate_reduction, tbCallBack])

# 不使用数据增强（数据增强训练速度过慢，不方便做实验）
# history = model.fit(
#     X_train,
#     Y_train,
#     epochs=nb_epoch,
#     batch_size=batch_size,
#     verbose=1,
#     validation_data=(X_val2, Y_val2),
#     callbacks=[learning_rate_reduction, tbCallBack])

# 评估
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# 保存结果
model.save('./model/my_model.h5')  # creates a HDF5 file 'my_model.h5'
plot_model(model, to_file='./model/model.png')
ModelUtil.saveHist('./model/file.json', history)
del model