# coding: utf-8
'''
__author__ = "Zhejian Peng"
__copyright__ = "Zhejian Peng"
__license__ = MIT LICENSE
__version__ = "1.0.1"
__maintainer__ = "Zhejian Peng"
__email__ = "zhejianpeng@yahoo.com"
__status__ = "Developing"

__update__ =
'''
# %%
import numpy as np
import pandas as pd

# %%
# Read Sugar data
sugar = pd.read_excel('Sugar Price.xlsx')
sugar.drop(['Unnamed: 2', 'Unnamed: 3'], axis=1, inplace=True)
# Read Russell 3000 daily return data
russell = pd.read_excel('EWMA-tf example (5).xlsx', sheetname='raw data')

# %% Use EWMA to model return and return^2, var = E[r^2] - E[r]^2


def ewma(numbda, ret):
    '''
    inputs:
        numbad: ewma constant
        ret: is the daily return
    output:
        return a array of ewma result
    '''
    n = len(ret)
    result = np.zeros(n)
    result[0] = (1 - numbda)*ret[0]
    for i in range(1, n):
        result[i] = (1-numbda)*ret[i] + numbda*result[i-1]
    return result


# %%
russell['return'] = ewma(
    0.9697, russell['Russell 3000 future contract daily return'])
russell['return^2'] = ewma(
    0.9697, russell['Russell 3000 future contract daily return']**2)
russell['devol_return'] = russell.iloc[:, 0] / \
    np.sqrt(russell.iloc[:, 2] - russell.iloc[:, 1]**2)

# %%
# Assume that the devolotized return follows t distribution
# We use MLE to estimate t distribution constant
# from scipy.special import gamma
from scipy.special import gamma
from scipy.stats import t
from scipy import optimize
import matplotlib.pyplot as plt
plt.hist(russell['devol_return'], bins=100)
# print(np.mean(russell['devol_return']))

# %%


def loc_scale_t(x, para):
    mu, sigma, v = para
    temp1 = gamma((v+1) / 2) / (sigma*np.sqrt(v*np.pi)*gamma(v/2))
    temp2 = ((v + ((x-mu)/sigma)**2) / v)**(-(v+1)/2)
    ret = temp1 * temp2
    return ret


# print(t.pdf(0, 5))
# para = [0, 1, 5]
# print(loc_scale_t(0, para))


def t_logL(para):
    ret = 1
    for x in russell['devol_return']:
        # print(loc_scale_t(x, para))
        ret = ret + np.log(loc_scale_t(x, para))
    return -ret


# %%
# t_logL([0.0399, 0.9151, 8.2777])
optimized_para = optimize.fmin(t_logL,  np.array([0.02, 0.2, 10]))
print('The optimized mu, sigma, and v is:', optimized_para)
print('MatLab Result is:',    [0.0406, 0.9056, 7.3312])

# %%
# Use Optimized t distribution to find upper 99.5% and lower 0.5% tail VaR.
mu, sigma, v = optimized_para
upper_tail = t.ppf(0.005, v, loc=mu, scale=sigma)
lower_tail = t.ppf(0.995, v, loc=mu, scale=sigma)
print('Upper_tail:', upper_tail, 'Lower_tail', lower_tail)
risky_tail = max(upper_tail, lower_tail)
print('The larger tail is', risky_tail)

# %%
# Compute return, settlement, and suggest margin
ret = risky_tail * \
    np.sqrt(russell['return^2'].iloc[-1] - russell['return'].iloc[-1]**2)
last_settlement = 664  # This is provide by the exchange
settlement_margin = ret*last_settlement*100
print('settlement_margin is:', settlement_margin)
# %%
sugar.tail()
# russell['devol_return'].iloc[-1]
