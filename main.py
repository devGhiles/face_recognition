import sys
import pickle
from data_reader import DataReader


# if the user didn't specify exactly one image path, we exit the program
if len(sys.argv) != 2:
    print('1 argument expected, found {0}'.format(len(sys.argv) - 1))
    exit()


# read the image
image_path = sys.argv[1]
data_reader = DataReader()
img = data_reader.read(image_path)


# feature extraction
# load the feature extractor
f = open('feature_extractor.pkl', 'rb')
feature_extractor = pickle.load(f)
f.close()

# extract the features from the image
X = feature_extractor.transform([img])


# image classification
# load the trained classifier
f = open('clf.pkl', 'rb')
clf = pickle.load(f)
f.close()

# predict the class (person)
y = clf.predict(X)[0]

# show the prediction
print('y = {0}'.format(y))
res = data_reader.read('data/s{0}/1.pgm'.format(y))
img.show()
res.show()