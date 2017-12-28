import os
import random
import matplotlib.pyplot as plt
from scipy import misc
from image import Image


class DataReader:
    """Reads the images from the database and splits them into
    train, validation and test images for classification purposes.
    """
    def __init__(self, data_path='data/', valid_ratio=0.0, test_ratio=0.0):
        self.valid_ratio, self.test_ratio = valid_ratio, test_ratio
        self.data, self.train_data, self.valid_data, self.test_data = ([], []), ([], []), ([], []), ([], [])

        # the data file structure is like this:
        # data
        #   |__ s1
        #        |__ 1.pgm
        #        |__ 2.pgm
        #        .
        #        .
        #        .
        #        |__ 10.pgm
        #   |__ s2
        #   .
        #   .
        #   .
        #   |__ s40
        # the folder "si" contains 10 images of the person "i"

        # class_folder will iterate through ['s1', 's2', ..., 's40']
        for class_folder in os.listdir(data_path):
            # each folder "si" corresponds to the person "i", so the label associated with
            # the current person is "i" ; that's what current_label captures
            current_label = int(class_folder.split('s')[1])

            # we know that we have 10 images for each person, we iterate through them
            for i in range(1, 11):
                # read the i-th image of the person
                img = self.read(data_path + class_folder + os.sep + "{0}.pgm".format(i))

                # update data
                self.data[0].append(img)
                self.data[1].append(current_label)

                # randomly decide to add the image the the train, validation, or test set,
                # depending on the ratio of each subset.
                r = random.random()
                if r < self.valid_ratio:
                    self.valid_data[0].append(img)
                    self.valid_data[1].append(current_label)

                elif r < self.valid_ratio + self.test_ratio:
                    self.test_data[0].append(img)
                    self.test_data[1].append(current_label)

                else:
                    self.train_data[0].append(img)
                    self.train_data[1].append(current_label)


    def getAllData(self, shuffle=False):
        """returns a tuple (images, labels), which is the content of self.data
        By default, the images are sorted according to the persons, meaning we
        have the 10 images of the first person (and the corresponding labels), followed
        by the 10 images of the second person, and so on.
        We might want to have the images randomly ordered in the dataset instead of being
        grouped by label. In order to acheive that, we shuffle the list of images, and we shuffle
        the list of labels in the same manner (the same re-ordering).
        """
        if not shuffle:
            return self.data

        # we need to shuffle
        # we suffle the indices
        shuffled_indexes = list(range(len(self.data[0])))
        random.shuffle(shuffled_indexes)

        # we re-order the data according the the suffled indices
        res = ([None for _ in range(len(self.data[0]))], [None for _ in range(len(self.data[1]))])
        for i, v in enumerate(shuffled_indexes):
            for j in range(2):
                res[j][i] = self.data[j][v]
        return res

    def getTrainData(self):
        return self.train_data

    def getValidData(self):
        return self.valid_data

    def getTestData(self):
        return self.test_data
        
    def read(self, img_path):
        return Image(misc.imread(img_path))
