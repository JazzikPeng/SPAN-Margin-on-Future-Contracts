# coding: utf-8
'''
# def a function to calcualte sugar margin
__author__ = "Zhejian Peng"
__copyright__ = "Zhejian Peng"
__license__ = MIT LICENSE
__version__ = "1.0.1"
__maintainer__ = "Zhejian Peng"
__email__ = "zhejianpeng@yahoo.com"
__status__ = "Developing"
__update__ =
'''
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import t
from scipy import optimize


class future:

    '''Every future contract traded in exchange need a margin.
    This class takes daily prices of a future contract.
    margin() function calculate the suggested margin of a future contract
    '''

    def __init__(self, prices, settlement, multiplier=100):
        '''
            Return a future object with prices and last settlement amount

            prices: a list of prices. eg. [11.2,11.3 ...]
            settlement: a float of last settlement amount
            multiplier: typical contract have a multiplier of 100
        '''
        self.prices = prices
        self.settlement = settlement
        self.multiplier = multiplier

    def get_prices(self):
        return self.prices

    def get_last_settlement(self):
        return self.settlement

    def daily_return(self):
        prices = self.prices
        log_return = []
        for i in range(1, len(prices)):
            temp = np.log((prices[i] / prices[i-1]))
            log_return.append(temp)
        return log_return

    def ewma(self, ret, numbda=0.9697):
        '''
        inputs:
            numbad: ewma constant
            ret: is the daily return
        output:
            return a array of ewma result
        '''
        # print('Type: ', type(ret))
        n = len(ret)
        # print('n', n)
        result = np.zeros(n)
        result[0] = (1 - numbda)*ret[0]
        for i in range(1, n):
            result[i] = (1-numbda)*ret[i] + numbda*result[i-1]
        return result

    def margin(self):
        # print('hi')
        daily_ret = self.daily_return()
        # daily_ret = self.prices
        # print('hi', len(daily_ret))
        # print(daily_ret)
        ret = self.ewma(daily_ret)
        ret_sqr = self.ewma([x**2 for x in daily_ret])
        devol_return = [
            x/np.sqrt(y-z) for x, y, z in zip(daily_ret, ret_sqr, [x**2 for x in ret])]

        def loc_scale_t(x, para):
            mu, sigma, v = para
            temp1 = gamma((v+1) / 2) / (sigma*np.sqrt(v*np.pi)*gamma(v/2))
            temp2 = ((v + ((x-mu)/sigma)**2) / v)**(-(v+1)/2)
            ret = temp1 * temp2
            return ret

        def t_logL(para):
            ret = 0
            # print(devol_return)
            for x in devol_return:
                # print(loc_scale_t(x, para))
                ret = ret + np.log(loc_scale_t(x, para))
            # print(-ret)
            return -ret

        optimized_para = optimize.fmin(t_logL,  np.array([0.02, 0.2, 10]))
        print('The optimized mu, sigma, and v is:', optimized_para)

        mu, sigma, v = optimized_para
        upper_tail = t.ppf(0.005, v, loc=mu, scale=sigma)
        lower_tail = t.ppf(0.995, v, loc=mu, scale=sigma)
        print('Upper_tail:', upper_tail, 'Lower_tail', lower_tail)
        risky_tail = max(upper_tail, lower_tail)
        print('The larger tail is', risky_tail)

        # Compute Margin
        temp = risky_tail * np.sqrt(ret_sqr[-1] - ret[-1]**2)
        # This is provide by the exchange
        last_settlement = self.get_last_settlement()
        multiplier = self.multiplier
        settlement_margin = temp*last_settlement*multiplier
        print('settlement_margin is:', settlement_margin)
        return settlement_margin


# Russell Future Margin
# russell_data = pd.read_excel('EWMA-tf example (5).xlsx', sheetname='raw data')
# russell_data = np.array(
#     russell_data['Russell 3000 future contract daily return'])
# # print(russell_data)
# russell = future(russell_data, 664)
# # print(russell.daily_return())
# russell.margin()

# Sugar Future Margin:
#%%
sugar_data = pd.read_excel('Sugar 11 Historical Prices.xls')
sugar_data = sugar_data[['DATE', 'CLOSE']]
# sugar_data
# sugar_data = sugar_data[['']]
# sugar_data.drop(['Unnamed: 2', 'Unnamed: 3'], axis=1, inplace=True)
sugar_data = np.array(sugar_data['CLOSE'])

# Here I dont know the last settlement, i used the last day's price to
# represent the settlement price
# %%
sugar = future(sugar_data, settlement=sugar_data[-1], multiplier=1120)
sugar.margin()
print(sugar.margin())
