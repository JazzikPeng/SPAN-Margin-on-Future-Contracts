# coding: utf-8
'''
Plot a implied volatility surface by estimating the implied volatiltiy using newton's method

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
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.optimize import newton
import datetime


def black_scholes(call_or_put, S0, K, r, T, sigma):
    d1 = (1/(sigma*np.sqrt(T)))*(np.log(S0/K)+(r+(sigma**2)/2)*T)
    d2 = d1-sigma*np.sqrt(T)
    if call_or_put.upper() == 'C':
        return S0*st.norm.cdf(d1)-K*np.exp(-r*T)*st.norm.cdf(d2)
    elif call_or_put.upper() == 'P':
        return K*np.exp(-r*T)*st.norm.cdf(-d2)-S0*st.norm.cdf(-d1)
    else:
        print('Please enter C for call option and P for put option')


# print(black_scholes('C', 10, 10, 0.1, 1/12, 0.5))


def find_implied_volatiltiy(call_or_put, mkt_price, S0, K, r, T):
    implied_sigma = newton(lambda x: black_scholes(
        call_or_put, S0, K, r, T, x)-mkt_price, 0.5)
    return implied_sigma


options = pd.read_csv("./Implied Volatility/C_U7P_Comdty.csv")
futures = pd.read_excel("./Implied Volatility/futures_price.xlsx")
futures_price = futures["Last Price"]

date = []
for i in range(len(options["Date"])):
    date.append(datetime.datetime.strptime(
        options["Date"][i], "%m/%d/%Y").date())

expiration = datetime.date(2017, 8, 25)

strike = np.linspace(310, 430, 13)

implied_vol = np.zeros((13, len(date)))
Time_2_maturity = np.zeros(len(date))

for j in range(len(date)):
    for i in range(len(strike)):
        implied_vol[i, j] = find_implied_volatiltiy('p', options[str(
            strike[i])+"0"][j],  futures_price[j], strike[i], 0.0105, (expiration-date[j]).days/360)
    Time_2_maturity[j] = (expiration-date[j]).days/360

plt.title("Implied Volatility Smile (2017-07-13)")
plt.plot(futures_price[0]/strike, implied_vol[:, 0])
plt.scatter(futures_price[0]/strike, implied_vol[:, 0])
plt.grid()
plt.xlabel("Moneyness")
plt.ylabel("Implied Volatility")
plt.show()
# black_scholes('P', 0, 0, 0, 0, 0)


# Volatility surface:
expiration = datetime.date(2017, 8, 25)

strike = np.linspace(310, 430, 13)
surface = np.zeros([len(strike), int(len(date))])
# print(surface.shape)
Y = np.zeros(int(len(date)))

for j in range(int(len(date))):
    for i in range(len(strike)):
        surface[i, j] = implied_vol[i, j]
    Y[j] = Time_2_maturity[j]


# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')

strike, Y = np.meshgrid(strike, Y)
ax.plot_surface(strike, Y, np.transpose(surface), cmap=cm.coolwarm)
# surf = ax.plot_surface(strike, Y, np.transpose(surface))
#     surface), cmap=cm.coolwarm, linewidth=0, antialiased=False)
# #fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_title("Implied Volatility Surface")
ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Implied Volatility')
plt.show()
