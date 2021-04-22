#!/usr/bin/env python3


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize

tau = 0.6
mu1 = -2.0
mu2 = 0.5
sigma1 = 0.4
sigma2 = 0.8
N = 10000
theta = np.array((tau, mu1, sigma1, mu2, sigma2))
x1 = np.random.normal(mu1, sigma1, size=(int(tau*N)))
x2 = np.random.normal(mu2, sigma2, size=(int((1-tau)*N)))
x = np.concatenate([x1, x2]) 


def f(x, tau, mu1, sigma1, mu2, sigma2):
    T1= tau /np.sqrt(2*np.pi*sigma1**2)*np.exp(-((x-mu1)**2)/(2*sigma1**2))
    T2= (1-tau) /np.sqrt(2*np.pi*sigma2**2)*np.exp((-(x-mu2)**2)/(2*sigma2**2))
    T = T1+T2
    
    T1 = np.divide(T1, T, where=T!=0, out = np.full_like(T1, 0.5) )
    T2 = np.divide(T2, T, where=T!=0, out = np.full_like(T2, 0.5) )

    return abs(T1), abs(T2)

def L(x, theta):
    tau, mu1, sigma1, mu2, sigma2 = theta[0], theta[1], theta[2], theta[3], theta[4]
    a1, a2 = f(x, tau, mu1, sigma1, mu2, sigma2)
    return -np.sum(np.log(abs(a1))+np.log(abs(a2)))


def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    theta = np.array([tau, mu1, sigma1, mu2, sigma2])
    ans = scipy.optimize.minimize(L, theta, args= x, tol = rtol, bounds = ((0, 1), (-np.inf, np.inf), (-np.inf, np.inf),
                                      (-np.inf, np.inf), (-np.inf, np.inf)))
    return ans.x


def th(x, tau, mu1, sigma1, mu2, sigma2):
    T1, T2 = f(x, tau, mu1, sigma1, mu2, sigma2)
    tau = np.sum(T1)/x.size
    mu1 = np.sum(T1*x)/np.sum(T1)
    mu2 = np.sum(T2*x)/np.sum(T2)
    sigma1 = np.sqrt(np.sum(T1*(x-mu1)**2)/np.sum(T1))
    sigma2 = np.sqrt(np.sum(T2*(x-mu2)**2)/np.sum(T2))
    return (tau, mu1, sigma1, mu2, sigma2)



def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    init_theta = tau, mu1, sigma1, mu2, sigma2
    current_theta = np.array((th(x, tau, mu1, sigma1, mu2, sigma2)))
    while np.allclose(init_theta, current_theta, rtol = rtol) == False:
        current_theta = th(x, *current_theta)
    return current_theta

def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-5):
    pass


if __name__ == "__main__":
    pass
