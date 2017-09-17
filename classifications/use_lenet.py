import numpy as np
from lenet import LeNet
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD
from keras.utils import np_utils
from sys import argv

print("[INFO] loading DATA...")
dataset = np.load(argv[1])
labelset = np.load(argv[2])

data = dataset[:, :, :, np.newaxis]
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labelset, test_size=0.2)

trainLabels = np_utils.to_categorical(trainLabels)
testLabels = np_utils.to_categorical(testLabels)

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet(width=dataset.shape[1], height=dataset.shape[2], depth=1, classes=len(np.unique(labelset)))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20, verbose=1)

print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("\n[INFO] accuracy: {:.2f}%".format(accuracy * 100))



