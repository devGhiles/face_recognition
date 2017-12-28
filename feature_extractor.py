import numpy as np


class FeatureExtractor:

    def __init__(self):
        pass

    def fit(self, images):
        # transform the 2D images into 1D vectors
        gammas = np.array([img.values.reshape((-1,)) for img in images])
        
        # the average image
        self.avg = gammas.mean(axis=0)

        # center the images
        phis = gammas - self.avg

        # eigenvectors of L = phis . phis'
        L = phis.dot(phis.T)
        v = np.linalg.eig(L)[1]

        # eigenvectors of the covariance matrix C = phis' . phis
        self.u = phis.T.dot(v)

        # normalize u
        self.u = (self.u - self.u.mean(axis=0)) / self.u.std(ddof=0)

        return self

    def transform(self, images):
        # transform the 2D images into 1D vectors
        gammas = np.array([img.values.reshape((-1,)) for img in images])

        # center the images
        phis = gammas - self.avg

        # return the coordinates in the facespace basis
        return phis.dot(self.u)

    def fit_transform(self, images):
        # transform the 2D images into 1D vectors
        gammas = np.array([img.values.reshape((-1,)) for img in images])
        
        # the average image
        self.avg = gammas.mean(axis=0)

        # center the images
        phis = gammas - self.avg

        # eigenvectors of L = phis . phis'
        L = phis.dot(phis.T)
        v = np.linalg.eig(L)[1]

        # eigenvectors of the covariance matrix C = phis' . phis
        self.u = phis.T.dot(v)
        
        # normalize u
        self.u = (self.u - self.u.mean(axis=0)) / self.u.std(ddof=0)

        # return the coordinates in the facespace basis
        return phis.dot(self.u)
