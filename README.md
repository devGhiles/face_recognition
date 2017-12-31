# Face recognition using eigenfaces and a naive bayes classifier

## Dependencies
The program is written in Python 3.6.1, see the requirements.txt file
for the required libraries. If you have pip installed, you can install
all the required libraries by typing the following command in the command-line:
```bash
$ pip install -r requirements.txt
```

## Description of the files
* __image.py__: contains the Image class, which is a class that wraps a 2D numpy array
and creates an abstraction for an image. It has several in-place methods such as
applyNoise and rotate.
* __data_reader.py__: contains the class DataReader. This class reads the images from the AT&T database,
wraps them using the Image class, and splits them into train, validation and test data. Also has
the read method which wraps the scipy.misc.imread method for reading an image file.
* __feature_extractor.py__: contains the class FeatureExtractor. This class implements 
the fit-transform paradigm. Given a list of images, it computes the eigenfaces and
saves them for future use. It also computes the coordinates on an image in the eigenfaces basis.
* __training.py__: generates the model by training a naive bayes classifier.
* __main.py__: recognizes the face in a new image using the model learned by training.py.
* __noise.py__: tests the robustness to noise of the system by applying different types of noises
to test images (gaussian, poisson, speckle, and salt & pepper).
* __contrast_inversion.py: tests the robustness to the contrast inversion of an image.
* __rotation.py: tests the robustness to the rotation of an image.
* __translation.py: test the robustness to the shifting of an image.

## Generate the model
The first step is to generate the model (which consists of training 
a naive bayes classifier). For that, you need to execute the file training.py:
```bash
$ python training.py
```

## Face recognition on a new image
The main.py file is used for that:
```bash
$ python main.py path/to/imagefile
```
Any image format that workds with scipy.misc.imread will work (that includes
'.jpg', '.png', and the more exotic '.pgm')

## Test the robustness of the system
This can be done either by altering only the test data or the training data and the
test data simultanously (see relevent source files for more details).

Noise:
```bash
$ python noise.py
```

Contrast inversion:
```bash
$ python contrast_inversion.py
```

Rotation:
```bash
$ python rotation.py
```

Translation (shifting):
```bash
$ python translation.py
```

## References
This work was inspired by [this paper](http://ieeexplore.ieee.org/abstract/document/7951418/).

We used the AT&T database of faces that you can find [here](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)