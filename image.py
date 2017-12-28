import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


class Image:
    """ An image abstraction class.
    values: numpy 2D array (we only handle grayscale images)
    """
    def __init__(self, values):
        # we save some general informations about the image
        self.values = values  # 2D numpy array
        self.height = values.shape[0]
        self.width = values.shape[1]
        self.size = (self.width, self.height)

    def show(self):
        plt.imshow(self.values)
        plt.show()

    def applyNoise(self, type='gaussian', **kwargs):
        """In-place method that applies a noise of given type
        and parameters on the image.
        type: 'gaussian', 'poisson', 'speckle', 's&p' (salt & pepper)
        kwargs:
            > mu: the expectancy of the gaussian noise
            > sigma: the variance of the gaussian noise
        """
        self.normalize(current_min=0, current_max=255, target_min=0.0, target_max=1.0, target_type=np.float)
        if type == 'gaussian' and False:
            # generate a gaussian noise and apply it to the image
            # set default values for mu and sigma if not given
            if 'mu' not in kwargs:
                kwargs['mu'] = 0.0
            if 'sigma' not in kwargs:
                kwargs['sigma'] = 0.1
            # add the noise
            noise = np.random.normal(loc=kwargs['mu'], scale=kwargs['sigma'], size=self.values.shape)
            self.values += noise

        self.values = random_noise(self.values, mode=type, seed=None, clip=True, **kwargs)

        # re-normalize (go back to [0, 255])
        current_max = max(1.0, self.values.max())
        current_min = min(0.0, self.values.min())
        self.normalize(current_min=current_min, current_max=current_max, target_min=0, target_max=255, target_type=np.int)

    def normalize(self, current_min=0, current_max=255, target_min=0.0, target_max=1.0, target_type=np.float):
        """change the range of pixel values from [current_min, current_max]
        to [target_min, target_max]. Eventually change the target type (integer or real).
        It's an in-place method (it changes the current image, it doesn't return the result).
        By default, it normalizes from [0, 255] to [0.0, 1.0].

        To achieve that, we apply an affine transformation: phi(x) = ax + b, where:
        a = (target_max - target_min) / (current_max - current_min)
        b = target_min - a * current_min
        """
        a = a = (target_max - target_min) / (current_max - current_min)
        b = target_min - a * current_min
        self.values = a * self.values + b
        self.values.astype(target_type)

    def copy(self):
        return Image(self.values.copy())