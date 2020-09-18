#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:51:30 2020

@author: albertzhu

this file contains numerical implementations for three risk measures: VaR, ES, put respectively and Gaussian Couplas in generating varying correlations between subportfolios.

"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.stats as ss
import scipy.optimize as opt

# Vasicek loss distribution

# Systematic factors S1 S2
N = 100000
S_a = np.random.normal(size = N)
S_b = np.random.normal(size = N)

rho1 = 0.4 # correlation to systamatic factor,  0 < rho <1
t1 = ss.norm.ppf(0.02) # default thresholds
# stipulated returns
g1 = 0.04

L1 = ss.norm.cdf((t1+np.sqrt(rho1)*S_a)/(np.sqrt(1-rho1)))
X1 = g1 - L1 # loans

sigma = 0.3
r = 0.02
m = 0.15
t = 1

#X1 = np.exp((m - 0.5 * sigma**2)*t + sigma * S_a * t**0.5) - 1
#X2 = np.exp((m - 0.5 * sigma**2)*t + sigma * S_b * t**0.5) - 1
L2 = ss.norm.cdf((t1+np.sqrt(rho1)*S_b)/(np.sqrt(1-rho1)))
X2 = g1 - L2

def func_var(w,X1,X2):
    X = w * X1 + (1-w) * X2
    
    mu_X = np.mean(X)
    mu_X1 = np.mean(X1)
    mu_X2 = np.mean(X2)
    
    Variance_X = np.var(X)
    
    VaR_X = np.quantile(-X,0.99)
    
    X_sort = np.sort(X)
    
    ULvar = mu_X + VaR_X
    
    beta1 = (np.dot((X - mu_X),(X1 - mu_X1))/len(X)) / Variance_X
    beta2 = (np.dot((X - mu_X),(X2 - mu_X2))/len(X)) / Variance_X
    
    # conditional VaRs
    VaR1 = beta1 * ULvar - np.mean(X1)
    VaR2 = beta2 * ULvar - np.mean(X2)
    
    RORAC = mu_X  / VaR_X

    return RORAC

def func_es(w,X1,X2):
    X = w * X1 + (1-w) * X2
    
    mu_X = np.mean(X)
    mu_X1 = np.mean(X1)
    mu_X2 = np.mean(X2)
    
    X_sort = np.sort(X)
    ES_X = -np.mean(X_sort[0:int(0.01 * N)])
   
    # conditional ESs
    ES1 = -np.mean(X1[np.argsort(X)][0:int(0.01*N)])
    ES2 = -np.mean(X2[np.argsort(X)][0:int(0.01*N)])
    

    RORAC = mu_X / ES_X

    return RORAC

def func_put(w,X1,X2):
    X = w * X1 + (1-w) * X2
    
    mu_X = np.mean(X)
    mu_X1 = np.mean(X1)
    mu_X2 = np.mean(X2)
    
    # put strike with risk free rate
    put = np.mean((0.02-X) * (X < 0.02))
   
    RORAC = mu_X / put

    return RORAC

corr = np.linspace(-1,0,11) # can chage to positive/negative
corr1 = np.zeros(len(corr))

wei = np.zeros(len(corr))
ress = np.zeros(len(corr))

wei2 = np.zeros(len(corr))
ress2 = np.zeros(len(corr))

wei3 = np.zeros(len(corr))
ress3 = np.zeros(len(corr))




for i in range(len(corr)):
    
    # Gaussian coupla methods to create correlations
    Norms = np.random.normal(size=(2,N))
    Norms[0] = corr[i] * Norms[1] + (1-corr[i]**2)**0.5 * Norms[0]

    Percentiles = ss.norm.cdf(Norms)

    Y1 = np.quantile(X1, Percentiles[0])
    Y2 = np.quantile(X2, Percentiles[1])

    corr1[i] = np.corrcoef(Y1,Y2)[0,1]
    
    obj = lambda w : -func_var(w, Y1,Y2) # change objective functions
    res = opt.minimize(obj,x0=(0.1),bounds = [(0,1)] )
    ress[i] = -res.fun
    wei[i] = res.x

for i in range(len(corr)):
    
    # Gaussian coupla methods to create correlations
    Norms = np.random.normal(size=(2,N))
    Norms[0] = corr[i] * Norms[1] + (1-corr[i]**2)**0.5 * Norms[0]

    Percentiles = ss.norm.cdf(Norms)

    Y1 = np.quantile(X1, Percentiles[0])
    Y2 = np.quantile(X2, Percentiles[1])

    corr1[i] = np.corrcoef(Y1,Y2)[0,1]
    
    obj = lambda w : -func_es(w, Y1,Y2) # change objective functions
    res = opt.minimize(obj,x0=(0.1),bounds = [(0,1)] )
    ress2[i] = -res.fun
    wei2[i] = res.x
    
for i in range(len(corr)):
    
    # Gaussian coupla methods to create correlations
    Norms = np.random.normal(size=(2,N))
    Norms[0] = corr[i] * Norms[1] + (1-corr[i]**2)**0.5 * Norms[0]

    Percentiles = ss.norm.cdf(Norms)

    Y1 = np.quantile(X1, Percentiles[0])
    Y2 = np.quantile(X2, Percentiles[1])

    corr1[i] = np.corrcoef(Y1,Y2)[0,1]
    
    obj = lambda w : -func_put(w, Y1,Y2) # change objective functions
    res = opt.minimize(obj,x0=(0.1),bounds = [(0,1)] )
    ress3[i] = -res.fun
    wei3[i] = res.x
    
    
plt.plot(corr1,wei,label="VaR")
plt.plot(corr1,wei2,label="ES")
plt.plot(corr1,wei3,label="put")

plt.legend()

plt.title('Optimal weight vs Correlation')
plt.xlabel("correlation")
plt.ylabel("optimal weight of subportfolio 1")
plt.show()



plt.plot(corr1,ress,label="VaR")
plt.plot(corr1,ress2,label="ES")


plt.legend()

plt.title("Optimal RORAC vs Correlation")
plt.xlabel("correlation")
plt.ylabel("optimal RORAC")
plt.show()

plt.plot(corr1,ress3,label="put")

plt.legend()

plt.title("Optimal RORAC vs Correlation")
plt.xlabel("correlation")
plt.ylabel("optimal RORAC")
plt.show()


