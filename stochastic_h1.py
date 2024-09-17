
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy import stats

from math import ceil
from typing import List

from simulation_operations import simulation_oper

month_of_birth1: int = 1
month_of_birth2: int = 1


time_span_months_unbound = 10 * (month_of_birth1 + month_of_birth2) - month_of_birth1 * month_of_birth2
time_span_months = min(max(time_span_months_unbound, 61), 120)

d2 = dt.datetime(2024, 8, 31)
d1 = d2 - relativedelta(months= time_span_months)

df = yf.download('^GSPC', start=d1.strftime("%Y-%m-%d"), end=d2.strftime("%Y-%m-%d"), interval = "1mo")
price_series = df['Close'].values

def get_estimates(price_series):
    """estimates for miu and sigma for simple returns """
    n_months = len(price_series)
    first_series = price_series[:n_months-1]
    second_series = price_series[1:]
    returns =  second_series/first_series - 1
    print(returns[:3])
    
    return np.mean(returns), np.std(returns)
 
def get_log_estimates(price_series):
    """estimates for miu and sigma for log returns """
    
    n_months = len(price_series)
    first_series = price_series[:n_months-1]
    second_series = price_series[1:]
    returns =  np.log(second_series/first_series)
    print(returns[:3])
    
    return np.mean(returns), np.std(returns)
   
 
def test_significance(miu, sigma, level, reference_miu = 0):
    """miu, sigma estimated for sample"""
    """ return true if it's outside the confidence interval for specified level """
    
    z_value = (miu - reference_miu)/sigma
    
    low  = st.norm.ppf(level/2, loc= 0, scale= 1)
    high = st.norm.ppf(1-level/2, loc= 0, scale= 1)
    
    return z_value > high or z_value < low
     

def simulate_forward_return(n_years, true_miu, true_sigma, n_simulations, return_type = "simple"):
    """true_miu, true_sigma are monthly"""
    """intermediate returns are stored only for simple returns"""
    n_months = n_years * 12
    
    def one_month_step(prev_price: float):
        if return_type == "simple":
            next_price = (np.random.normal(true_miu, true_sigma, 1)  + 1) * prev_price   
        else:
            next_price = np.exp(np.random.normal(true_miu, true_sigma, 1)) * prev_price   
            
        return next_price
    
    def multiple_months_steps(prev_price: float, n_steps: int):
        """ simulate n months price variation """
    
        if n_steps == 0:
            return []
        
        returns = []
        
        for _ in range(n_steps):
            next_price = one_month_step(prev_price)
            if return_type == "simple":
                returns.append(next_price/prev_price -1)
            prev_price = next_price
            
        return np.array(returns) if len(returns) > 0  else next_price
    
    #start from the same value as real data
    first_price = price_series[0] if return_type == "simple" else price_series[-1]
    first_simulation = multiple_months_steps(first_price, n_months)

    
    #each column is the arry of returns for a simulation
    for _ in range(n_simulations - 1):
        first_simulation = np.hstack((first_simulation, multiple_months_steps(first_price, n_months)))
        
    all_simulations = first_simulation
    
    return all_simulations

#requirement a 
miu, sigma = get_estimates(price_series)
print((miu, sigma))
print(f"significant miu {test_significance(miu, sigma, 0.05, 0)}")

#requirement b

margin_of_error = 0.1/200
true_sigma = 0.06
true_miu = 0.004

# margin_of_error 5% level = 1.96 * true_sigma / np.sqrt(n)

n = (1.96 * true_sigma/margin_of_error)**2
n_years = round(n/12)
print(f"{n_years} to achieve desired confidence interval for miu")


#requirement c

all_simulations = simulate_forward_return(n_years, true_miu, true_sigma, 20)

sim_obj = simulation_oper(all_simulations)

sim_passed = sim_obj.check_validity(true_miu - margin_of_error, true_miu + margin_of_error)

print(f"Number of percent simulations passed {sim_passed}")


#requirement d
miu_log, sigma_log = get_log_estimates(price_series)
print((miu, sigma))

#requirement e

#31 of August 2024
start_price = price_series[-1]
expected_forward = start_price * np.exp(miu_log)**60
print(f"Expected index in 5 years {expected_forward}")


#requirement f

all_log_simulations = simulate_forward_return(5, miu_log, sigma_log, 10000, return_type = "log")

#sturge rule for number of bins
plt.hist(all_log_simulations, bins=1 + ceil(np.log2(len(all_log_simulations))), edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Histogram of Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')

# Show the plot
plt.show()


statistic, p_value = stats.kstest(all_log_simulations, 'norm', args=(np.mean(all_log_simulations), np.std(all_log_simulations)))


#requirement g, h

#for options
split_price = 5600

put_prices = [1 for last_price in all_log_simulations if  last_price <= split_price]
call_prices = [1 for last_price in all_log_simulations if  last_price > split_price]

#Arrow security prices are all the same each simulation path is equally likely, their "price" is 1/10000
#that gives the below formulas  
put_price = np.sum(put_prices)/(np.sum(put_prices) + np.sum(call_prices))
call_price = 1 - put_price

print(f"put_price - call_price \n {put_price} - {call_price}")

#############  PART 2 ###########################################

#maturity 3 months 
#strike price for call option 5600 
#3 months interest rate 1% annual 4% 
#3 steps binomial tree 
#s0 = 31 august 2024 
#monthly net return 0.6% 
#p = 0.56

s0 = start_price

def build_binomial_state(s: int, state_code: List, max_level = 3, u = 1.006, d = 0.994):
    """ calculate the price of the index in this state define by state_code """  
    """ 1 in state code is an up movement, 0 is a down movement """
    
    up_movements = np.sum(state_code)
    down_movements = len(state_code) - up_movements
    
    
    return s * u**up_movements * d **down_movements


#requierement a
# after first time step from lowest to highest
s10 = build_binomial_state(s0, [0])
s11 = build_binomial_state(s0, [1])

#after second time step from lowest to highest
s20 = build_binomial_state(s0, [0, 0])
s21 = build_binomial_state(s0, [0, 1])
s22 = build_binomial_state(s0, [1, 1])

#after third time step from lowest to highest
s20 = build_binomial_state(s0, [0, 0, 0])
s21 = build_binomial_state(s0, [0, 0, 1])
s22 = build_binomial_state(s0, [0, 1, 1])
s23 = build_binomial_state(s0, [1, 1, 1])


#requirement b
#by q*s11 + (1- q)* s10 = s0
#  q*(s11-s10) + s10 = s0

#simple return for 3 months 
r = 0.01
q = (s0-s10)/(s11-s10)
terminal_prices = [s20, s21, s22, s23]
terminal_payoffs = [s if s > 5600 else 0 for s in terminal_prices]
terminal_probabilities_q = [q**3, q**2*(1-q), q*(1-q)**2, (1-q)**3]

price_call_option = np.dot(terminal_payoffs, terminal_probabilities_q)/(1 + r)
print(price_call_option)
# 



