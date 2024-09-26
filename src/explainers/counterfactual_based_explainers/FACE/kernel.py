from abc import abstractmethod
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.special import gamma as gamma_function


class Kernel:
    def __init__(
            self,
            data,
            bandwidth,
            volume,
            knnk,
            number_of_points,
            epsilon: float = 1e-8
    ):
        self.data = data
        self.bandwidth = bandwidth
        self.volume = volume
        self.knnk = knnk
        self.number_of_points = number_of_points
        self.eps = epsilon


    def __call__(
            self,
            xi,
            xj
    ):
        return self.func(xi, xj)
    
    @abstractmethod
    def func(
            self,
            xi,
            xj
    ):
        pass
    
class KDE(Kernel):
    def __init__(self, data, bandwidth, epsilon):
        self.data = data
        self.bandwidth = bandwidth
        self.epsilon = epsilon
        super(KDE).__init__()

    def func(self, xi, xj):
        mean = 0.5*(xi + xj)
        dist = np.linalg.norm(xi - xj, 2)
        kde = KernelDensity(kernel='gaussian', bandwidth= self.bandwidth).fit(self.data)
        density_at_mean = np.exp(kde.score_samples([mean]))
        return (1/(density_at_mean + self.eps))*dist   

    def __call__(self, xi, xj):
        return self.func(xi, xj)    
    
class KNN(Kernel):
    def __init__(self, data, volume, knnk, number_of_points):
        self.data = data
        self.volume = volume
        self.knnk = knnk
        self.number_of_points = number_of_points
        super(KNN).__init__()

    def func(self, xi, xj):
        dim = len(xi)
        if self.volume is None:
            self.volume = np.pi**(dim//2) / gamma_function(dim//2 + 1)
        dist = np.linalg.norm(xi - xj, 2)
        density_at_mean = self.knnk/(self.number_of_points*self.volume)*dist
        return density_at_mean

    def __call__(self, xi, xj):
        return self.func(xi, xj)