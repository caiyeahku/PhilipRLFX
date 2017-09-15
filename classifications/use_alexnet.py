from alexnet import AlexNet
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from sys import argv
import numpy as np

LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'simple-alexnet-epoch{}.model'.format(EPOCHS)

print("[INFO] loading DATA...")
dataset = np.load(argv[1])
labelset = np.load(argv[2])
WIDTH = dataset.shape[1]
HEIGHT = dataset.shape[2]

data = dataset[:, :, :, np.newaxis]
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labelset, test_size=0.1)

trainLabels = np_utils.to_categorical(trainLabels)
testLabels = np_utils.to_categorical(testLabels)

print("[INFO] compiling model...")
model = AlexNet(width=WIDTH, height=HEIGHT, lr=LR)

print("[INFO] training...")
model.fit({'input': trainData}, {'targets': trainLabels},
            validation_set=0.1, n_epoch=EPOCHS, snapshot_step=500, show_metric=False, run_id=MODEL_NAME)

print("[INFO] evaluating...")
accuracy = model.evaluate(testData, testLabels)[0]
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

