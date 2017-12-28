import numpy as np
import math
import random
from data_reader import DataReader
from feature_extractor import FeatureExtractor
from sklearn.naive_bayes import GaussianNB


data_reader = DataReader()  # reads the images files and converts them into numpy 2D arrays
feature_extractor = FeatureExtractor()  # calculates the eigenfaces. Follows the fit->transform paradigm.
clf = GaussianNB()  # a naive bayes classifier where the individual variables are supposed to follow a gaussian distribution


# since the number of images available is relatively low (400 images),
# we'll use cross-validation to assess the performance of the face recognition system.
data = data_reader.getAllData(shuffle=True)  # we shuffle the data so we can do Cross-Validation
# uncomment the 3 lines below to reverse the contrast of some (30%) of the images
# for img in data[0]:
#     if random.random() < 0.3:
#         img.reverseContrast()

num_folds = 10
fold_length = math.floor(len(data[0]) / num_folds)
average_accuracy = 0.0  # the performance measure of the system

for k in range(num_folds):
    # get train data and test data from data
    train_data, test_data = [None, None], [None, None]
    for i in range(2):
        if k == num_folds - 1:
            train_data[i] = data[i][:k * fold_length]
            test_data[i] = data[i][k * fold_length:]

        else:
            train_data[i] = data[i][:k * fold_length] + data[i][(k + 1) * fold_length:]
            test_data[i] = data[i][k * fold_length:(k + 1) * fold_length]
    train_data, test_data = tuple(train_data), tuple(test_data)

    # reverse the contrast of all the test images
    for i, img in enumerate(test_data[0]):
        reversed_img = img.copy()
        reversed_img.reverseContrast()
        test_data[0][i] = reversed_img

    # compute the eigenfaces and prepare the training data to train the classifier
    X_train = feature_extractor.fit_transform(train_data[0])  # computes eigenfaces and prepares training data
    y_train = np.array(train_data[1])  # prepares training labels
    clf.fit(X_train, y_train)  # trains the classifier

    # test the performance (accurancy) of the classifier on the current fold
    X_test = feature_extractor.transform(test_data[0])  # prepares the test data
    y_test = np.array(test_data[1])  # prepares the test labels
    average_accuracy += clf.score(X_test, y_test)  # accumulates the accuracies on each fold, we'll divide it by the number of folds later

average_accuracy /= num_folds  # computes the average accuracy of the classifier over all the folds

print('Average accuracy: {0}%'.format(round(100 * average_accuracy), 2))