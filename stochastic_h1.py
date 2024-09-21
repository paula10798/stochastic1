import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import math
import statsmodels.api as sm

from math import ceil
from typing import List

# Calculating the sample size
month_of_birth1: int = 3
month_of_birth2: int = 9
sample_size = 10 * (month_of_birth1 + month_of_birth2) - (month_of_birth1 * month_of_birth2)
print("The sample size is", sample_size)

d2 = dt.datetime(2024, 8, 31)
d1 = d2 - relativedelta(months= sample_size)

# Downloading the data 
df = yf.download('^GSPC', start=d1.strftime("%Y-%m-%d"), end=d2.strftime("%Y-%m-%d"), interval = "1mo")
price_series = pd.DataFrame(df['Close'])


#---------------------------------------------------------------------------#
########### QUESTION 1 ######################################################
#---------------------------------------------------------------------------#

## Requirement a 

# Calculating simple net returns 
price_series['Returns'] = price_series['Close'].pct_change()

# Estimating mu
mu_estimate = price_series['Returns'].mean()
# Estimating sigma 
sigma_estimate = price_series['Returns'].std()

# Testing procedure and conclusion
n = len(price_series)
significance_level = 0.05
cv_1 = st.norm.ppf(significance_level/2, loc= 0, scale= 1)     # Lower critical value of standard normal distribution
cv_2 = st.norm.ppf(1 - significance_level/2, loc= 0, scale= 1) # Upper critical value of standard normal distribution
mu_0 = 0  # The null hypothesis

t_stat = (mu_estimate - mu_0) / (sigma_estimate / math.sqrt(n)) 
                                 
print(t_stat > cv_2 or z_value < cv_1)  # Conclusion is: True
# Therefore we can reject the null hypothesis that mean returns are equal to 0  

#---------------------------------------------------------------------------#
                                 
## Requirement b

# Defining the true parameters
margin_of_error = 0.1/200
true_sigma = 0.06
true_mu = 0.004

# Following formula can be applied to help calculate the required number of years of return data 
# margin_of_error at 5% level = 1.96 * true_sigma / np.sqrt(n)

# Calculating the number of years
months_required = (1.96 * true_sigma/margin_of_error)**2
years_required = round(months_required/12)
print(f"{years_required} years of data is approximately required to achieve the desired confidence interval for mu")

#----------------------------------------------------------------------------#

## Requirement c

# Simulation exercise
n_months = years_required * 12
simulation_exercise = np.zeros((n_months,1))

for t in range(1, n_months):
    epsilon = np.random.normal(0, 1)
    simulation_exercise[t] = true_mu + ( true_sigma * epsilon)

mu_simulation = simulation_exercise.mean()
print(f"The simululated mean (mu = {mu_simulation}) falls within the confidence interval [0.35%, 0.45%]")

#-----------------------------------------------------------------------------#

## Requirement d

# Calculating log returns
price_series['Log Returns'] = np.log(price_series['Close'] / price_series['Close'].shift(1))
mean_log_returns = price_series['Log Returns'].mean()
std_log_returns = price_series['Log Returns'].std()

# Estimating sigma, via an OLS regression
price_series.dropna(inplace = True)
price_series['Constant']
y = price_series['Log Returns']
X = price_series['Constant']
model = sm.OLS(y, X).fit()

tilde_sigma = np.std(model.resid, ddof=1)

# Estimating mu
constant = model.params['Constant']
tilde_mu = constant + ((1/2) * (tilde_sigma**2))

print(f"Tilde mu is {tilde_mu} while tilde sigma is {tilde_sigma}")

#-----------------------------------------------------------------------------#

## Requirement e

# Calculating the analytical expectation
start_price = price_series.iloc[-1, 0]  #31 of August 2024
expected_forward = start_price * np.exp(constant * 60)
print(f"Expected index in 5 years {expected_forward}")

expected_forward_v2 = start_price * np.exp(mean_log_returns * 60)  # Which calculates the same expected price 

#-----------------------------------------------------------------------------#

## Requirement f

# Simulating the stock prices 1000 times 
log_simulations_1000 = np.zeros((61, 1000))
log_simulations_1000[0, :] = start_price

for i in range(1000):
    for t in range(1, 61):
        epsilon = np.random.normal(0, 1)
        log_simulations_1000[t, i] = log_simulations_1000[t-1 , i] * np.exp((mean_log_returns - 0.5  * std_log_returns**2) + std_log_returns * epsilon)

log_simulations_1000 = pd.DataFrame(log_simulations_1000)

# Distribution plot and shape
plt.hist(log_simulations_1000[1:])
plt.title('Distribution of Simulated Stock Prices')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.show()

# Testing procedure and conclusion
array_1d = log_simulations_1000.to_numpy().flatten()
statistic, p_value = stats.kstest(array_1d, 'norm', args=(np.mean(array_1d), np.std(array_1d)))
print(statistic)  # p-value = 0.1146

# Therefore, we do not reject the null hypothesis that the simulated stock returns follow a normal distribution. 

#-----------------------------------------------------------------------------#

## Requirement g

strike_price = 5600

# Digital put payoffs
S_T = log_simulations_1000.iloc[:, -1]
expected_payoff_digital_put = np.where(S_t < strike, 1, 0)

# Digital put price
digital_put_price = np.mean(expected_payoff_digital_put)
print(f"The price of the digital put is {digital_put_price}")

#---------------------------------------------------------------------------#

## Requirement h

# Digital call price
expected_payoff_digital_call = np.where(S_T > strike_price, 1, 0)
digital_call_price = np.mean(expected_payoff_digital_call)
print(f"The price of the digital put is {digital_call_price}")

# Sum of put and call
total_put_call = digital_call_price + digital_put_price
print(f"The sum of the put and call is {total_put_call}")

#------------------------------------------------------------------------#
########### QUESTION 2 ###################################################
#------------------------------------------------------------------------#

## Requirement a

# Initial parameters provided
mu = 0.006
p = 0.56
variance = price_series['Returns'].var() 
N = 3                   # Number of time steps 

# Calculating u and d - by constructing a system of two equations, using the expectation and variance
from sympy import symbols, Eq, solve
r_u, r_d = symbols(' r_u r_d')
eq1 = Eq((p * r_u) + ((1 - p) * r_d), mu)
eq2 = Eq((p*(r_u - mu)**2) + ((1-p) * (r_d - mu)**2), variance)
solutions = solve((eq1, eq2) , (r_u, r_d))

## Two sets of solutions: 
## (0.0476996520990794, -0.0470722844897374) 
## (-0.0356996520990794, 0.0590722844897374)
# Storing the first st of solutions
solution_dict = dict(solutions[0])
r_u = solution_dict[r_u]
r_d = solution_dict[r_d]

u = 1 + r_u
d = 1 - r_d
      
# Verifying the conditions
expected_return = (r_u * p) + ((1-p) * r_d)
expected_variance = (p * (r_u - mu)**2) + ((1-p) * (r_d - mu)**2)

# Constructing the tree with the stock prices 
def binomial_model(S, u, d, N):
    stock_tree = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(i+1):
            stock_tree[j, i] = S * (u **(i-j)) * (d ** j)
    return stock_tree

binomial_model_stock = binomial_model(start_price, u, d, N)
print(binomial_model_stock)

# Plotting the binomial tree
import pyop3  # Note... first download this package: pip install -i https://test.pypi.org/simple/ PyOptionTree
pyop3.tree_planter.show_tree(binomial_model_stock, "Binomial Tree Price Development of Underlying Asset")

#----------------------------------------------------------------------#

## Requirement b

r_f = 0.01
strike_price = 5600

# Calculating the risk-neutral probabilities 
q_u = ((1 + r_f) - d) / ( u - d)  # upward state probability
q_d = 1 - q_u                     # downward state probability

# Calculating the payoff 
risk_neutral_payoff = (binomial_model_stock[0,1] * q_u) + (binomial_model_stock[1,1] * q_d)
risk_neutral_return = (risk_neutral_payoff / start_price) - 1 
print(f"Applying the risk neutral probabilities in order to calculate the expected return, we find that the risk neutral return, {risk_neutral_return}, is equal to the risk-free rate, {r_f}")

# Option price of the European Call option
call_option_tree = np.zeros((N+1, N+1))
for i in range(0,N):
    call_option_tree[i, -1] = max(binomial_model_stock[i, -1] - strike_price, 0)

# Need to figure out how these manual calculations can be performed more efficiently 
call_option_tree[0,2] = (1 / (1 + r_f)) * ((q_u * call_option_tree[0,3]) + (q_d * call_option_tree[1,3]))
call_option_tree[1,2] = (1 / (1 + r_f)) * ((q_u * call_option_tree[1,3]) + (q_d * call_option_tree[2,3]))
call_option_tree[2,2] = (1 / (1 + r_f)) * ((q_u * call_option_tree[2,3]) + (q_d * call_option_tree[3,3]))
call_option_tree[0,1] = (1 / (1 + r_f)) * ((q_u * call_option_tree[0,2]) + (q_d * call_option_tree[1,2]))
call_option_tree[1,1] = (1 / (1 + r_f)) * ((q_u * call_option_tree[1,2]) + (q_d * call_option_tree[2,2]))
call_option_tree[0,0] = (1 / (1 + r_f)) * ((q_u * call_option_tree[0,1]) + (q_d * call_option_tree[1,1]))

call_option_price = call_option_tree[0,0]
print(f"The price of the call option should be {call_option_price}")

pyop3.tree_planter.show_tree(call_option_tree, "Binomial Tree Price Development of the European Call Option")

#----------------------------------------------------------------------------#

## Requirement  c

Nm = norm.cdf
                 
def BS_call_price(S_0, strike_price, sigma, T, r_f):
    d1 = (np.log(S_0 / strike_price) + (r_f + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S_0 * Nm(d1) - strike_price * np.exp(-r_f * T)* Nm(d2)

S_0 =
sigma = 
BS_call_option_price = BS_call_price(S_0, strike_price, sigma, T, r_f)

print(f"The price of the call option, using the Black Scholes pricing model, is {BS_call_option_price}")

#-----------------------------------------------------------------------------#

## Requirement d

#-----------------------------------------------------------------------------#

## Requierement e , the sign of expected return if simulation with p is higher than simulation with q ? 

#-----------------------------------------------------------------------------#

## Requirement f  what is call-put parity 

#-----------------------------------------------------------------------------#

## Requirement g replication method ?? 

