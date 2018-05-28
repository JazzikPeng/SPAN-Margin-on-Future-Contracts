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


print('Hello')
