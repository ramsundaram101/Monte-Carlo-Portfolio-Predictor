import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def str2bool(v): #Converts string to bool for argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Monte Carlo on stocks')
parser.add_argument('--percent', type = str2bool, default=True, help = 'False in order to output actual, not percent returns')
parser.add_argument('--compare_market', type = str2bool, default=False, help = 'Mark True to also factor in market state to portfolio returns')
parser.add_argument('--alpha', type = int, default=95, help = 'The confidence interval upto which you want to predict VaR and cVaR')
args = parser.parse_args()

#Gets current price of a particular stock ticker
def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]

#Need to add how to deal with stocks that weren't listed before start of lookback_period

#Gets the data for a list of stocks from start to end date. Returns percent change and covariance Matrix
def getData(stocks, start, end):
    stockData = yf.download(stocks, start, end)['Close']
    returns = stockData.pct_change().dropna()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

#Calculates weight of stocks in a portfolio given no. of each stock
def calcWeights(stocks, stock_nos):
  stock_val = np.zeros(shape = (len(stocks)), dtype = np.float32)
  for ind, stock in enumerate(stocks):
    stock_val[ind]=(get_current_price(stock)*stock_nos[ind])
  total_val = np.sum(stock_val)
  return [stock/total_val for stock in stock_val], total_val

#Runs the actual Monte Carlo Simulation
def MonteCarlo(weights, meanReturns, total_val, covMatrix, T = 100, mc_sims = 400):
  meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns) #Nothing but the avg returns for each stock
  meanM = np.transpose(meanM)
  portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0) #Here, we will add portfolio results for m simulations over T time periods
  portfolio_return_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
  #For loop running the actual sims
  for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights))) #Each stock is assumed a normal variable to daily returns are a random walk through normal distribution
    L = np.linalg.cholesky(covMatrix) #We use a Cholesky Decomposition to Generate our Multivariate Normal Distribution. 
    dailyReturns = meanM + np.inner(L, Z) 
    '''
    Daily returns are nothing but the Mean +- the Inner product of Separate Normal Distributions and the Cholesky Lower Triangle Matrix(L)
    Here, this inner product is what generates our Multivariate Normal Distribution.
    For the purposes of this Monte Carlo SIm, we assume portfolio returns to be a Multivariate Normal Distribution and each simulation is a random walk through the same.
    For more details, refer http://www.math.kent.edu/~reichel/courses/monte.carlo/alt4.7c.pdf
      '''
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*total_val #Multiplying by weights and initial portfolio value to get our returns
    portfolio_return_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*100

  return portfolio_sims, portfolio_return_sims
def VaR(returns, alpha=95):
  return np.percentile(returns, 100 - alpha)
    
def cVaR(returns, alpha=95):
  belowVaR = returns <= VaR(returns, alpha)
  return returns[belowVaR].mean()

if __name__ == '__main__':
    percent = args.percent
    compare_market = args.compare_market
    alpha = args.alpha
    
    #First taking user input
    no_stocks = int(input('Please enter the number of stocks: '))
    
    stockList = []
    stock_nos = []
    for i in range(no_stocks):
      stock = input('Please Enter NSE Ticker of Stock {:d}: '.format(i+1))
      stockList.append(stock)
      no = int(input('Please Enter No. of shares of {:s} you own: '.format(stock)))
      stock_nos.append(no)
    lookback_period = float(input('Please enter the lookback period you wish to use data from (yrs, can be a decimal): '))
    mc_sims = int(input('Please enter the number of simulations you wish to run(try to keep under 100,000): '))
    T = float(input('Please enter the number of yrs into the future you want to predict portfolio returns (yrs, can be decimal): '))
    
    #Making a few rudimentary changes to the input for processing like converting days to years
    stocks = [st + '.NS' for st in stockList]
    lookback_period = int(lookback_period*365)
    T = int(T*365)
    if len(stockList)==1 or compare_market: #If only 1 stock or compare_market is True, compare our stock to 0 shares of NIFTY as market 
      stocks.append('^NSEI') 
      stock_nos.append(0)
        
    end = dt.date.today() #Sets today's date as end date
    start = end - dt.timedelta(days = lookback_period)
    #Calling our helper/preprocessing functions
    returns, meanReturns, covMatrix = getData(stocks, start, end)
    weights, total_val = calcWeights(stocks, stock_nos)
    portfolio_vals, portfolio_percents = MonteCarlo(weights, meanReturns, total_val, covMatrix, T, mc_sims) #Runs the sims
    if(percent): #Portfolio returns in absolute or percentage terms
      results = portfolio_percents[-1, :]
    else:
      results = portfolio_vals[-1, :]
    
    #SNS Plot for our simulated returns
    sns.set_style('darkgrid')
    ax = sns.histplot(data=results, kde=False, stat='density', bins = 35, fill=True, color = '#97BC62FF')
    sns.kdeplot(data=results, color='#00203FFF', ax=ax)
    kde_x, kde_y = ax.lines[0].get_data()
    x0 = VaR(results, alpha)
    ax.fill_between(kde_x, kde_y, where=(kde_x<x0) , interpolate=True, color = 'red')
    if(not percent):
      ax.set_title("Probability Distribution of Your Portfolio Returns")
    else:
      ax.set_title("Probability Distribution of Your Portfolio Percentage (%) Returns")
    ax.set(xlabel='Portfolio Returns', ylabel='Likelihood')
    ax.set_yticklabels([])
    
    #Printing portfolio simulation results and showing probability plot
    print("Initial Portfolio Value: Rs. {:d}".format(int(total_val)))
    if(not percent):
      print("25th Percentile Returns: Rs. {:.2f}".format(np.percentile(results, 25)))
      print("Average (50th Percentile) Returns: Rs. {:.2f}".format(np.percentile(results, 50)))
      print("75th Percentile Returns: Rs. {:.2f}".format(np.percentile(results, 75)))
      print("Variance at Risk: Rs. {:.2f}".format(VaR(results)))
      print("Conditional Variance at Risk: Rs. {:.2f}".format(cVaR(results)))
    else:
      print("25th Percentile Returns: {:.2f} %".format(np.percentile(results, 25)))
      print("Average (50th Percentile) Returns: {:.2f} %".format(np.percentile(results, 50)))
      print("75th Percentile Returns: {:.2f} %".format(np.percentile(results, 75)))
      print("Variance at Risk: {:.2f} %".format(VaR(results)))
      print("Conditional Variance at Risk: {:.2f} %".format(cVaR(results)))
    plt.show()