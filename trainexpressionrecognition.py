# -*- coding: utf-8 -*-

# 导入包
from cnnnet import CNNNet
from oldcare.preprocessing import SimplePreprocessor
from oldcare.datasets import SimpleDatasetLoader
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# 全局变量
dataset_path = 'dataset/expdataset'
accuracy_plot_path = 'plots/accuracy.png'
loss_plot_path = 'plots/loss.png'
output_model_path = 'models/face_expression.hdf5'

# 全局常量
TARGET_IMAGE_WIDTH = 48
TARGET_IMAGE_HEIGHT = 48
NUM_CLASSES = 2
LR = 0.001  # 学习率
BATCH_SIZE = 64
EPOCHS = 60

################################################
# 第一部分：数据预处理

# initialize the image preprocessor and datasetloader
sp = SimplePreprocessor(TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)
sdl = SimpleDatasetLoader(preprocessors=[sp])

# Load images
print("[INFO] 导入图像...")
image_paths = list(paths.list_images(dataset_path))  # path included
(X, y) = sdl.load(image_paths, verbose=500, grayscale=True)

print(X.shape)
print(y.shape)

# Show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB"
      .format(X.nbytes / (1024 * 1024.0)))

# Label encoder
le = LabelEncoder()
y = to_categorical(le.fit_transform(y), NUM_CLASSES)
print(le.classes_)

# 拆分数据集
(trainData, testData, trainLabels, testLabels) = train_test_split(X, y, test_size=0.2, random_state=0)

# matrix shape should be: num_samples x rows x columns x depth
trainData = trainData.reshape((trainData.shape[0], TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT, 1))
testData = testData.reshape((testData.shape[0], TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT, 1))

# scale data to the range of [0,1]
trainData = trainData.astype('float32') / 255.0
testData = testData.astype('float32') / 255.0

################################################
# 第二部分：创建并训练模型
# initialize the optimizer and model
print('[INFO] 编译模型...')
# opt = SGD(lr=LR)
opt = Adam(lr=LR)
# opt = RMSprop(lr=LR)
model = CNNNet.build(TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT, NUM_CLASSES, '', 1)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

# train model
print('[INFO] 训练模型...')
H = model.fit(trainData, trainLabels,
              validation_data=(testData, testLabels),
              batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

################################################
# 第三部分：评估模型

# 画出accuracy曲线
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, EPOCHS + 1), H.history["acc"], label="train_acc")
plt.plot(np.arange(1, EPOCHS + 1), H.history["val_acc"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(accuracy_plot_path)

# 画出loss曲线
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, EPOCHS + 1), H.history["loss"], label="train_loss")
plt.plot(np.arange(1, EPOCHS + 1), H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(loss_plot_path)

# 打印分类报告
# show accuracy on the testing set
print('[INFO] 评估模型...')
predictions = model.predict(testData, batch_size=32)
print(classification_report(testLabels.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(i) for i in range(NUM_CLASSES)]))

################################################
# 第四部分：保存模型
model.save(output_model_path)
