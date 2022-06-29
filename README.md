# Monte Carlo Portfolio Predictor
Monte Carlo Simulation using Historical Data to Predict Portfolio Returns over a Given Period

## The Model
This model uses Monte Carlo Simulations (basically, running a large number of "random walks") to predict portfolio returns. Our Monte Carlo model assumes the distribution of daily returns of a portfolio to be a MultiVariate Normal Distribution. 
This enables us to perform a random walk through our distribution i.e picking daily stock price movements on basis of their relative probability and run simulations accordingly

The mechanism through which this is accomplished is slightly more complex but boils down to something called a Cholesky Decomposition which enables us to extract probabilities from a MultiVariate Distribution. 

Find a better explanation and more details on this here : http://www.math.kent.edu/~reichel/courses/monte.carlo/alt4.7c.pdf

## VaR and CVaR
Value at Risk and Conditional Value at Risk are two metrics commonly used in the world of risk management. Our model additionally calculates these.
VaR basically tells us, upto a certain confidence level (usually 95%), what is the maximum money we could expect to lose i.e our maximum possible risk.

CVaR, also known as the expected shortfall, then tells us how much money we might expect to lose given a very extreme situation, a situation beyond our wildest nightmares. Nightmare is just pompous talk for our previously mentioned confidence level.

For more details,
*  VaR : https://www.investopedia.com/articles/04/092904.asp
* CVaR : https://www.investopedia.com/terms/c/conditional_value_at_risk.asp

## Sample Results
Let's now run our model and look at some results. To find the actual code needed, scroll down a bit.
We'll predict portfolio performance for a portfolio consisting of Reliance, TCS, ITC, Maruti and Infosys. Do note that you have to input NSE Tickers for the model to work.

#### Inputs

Here's our input


![image](https://user-images.githubusercontent.com/87599801/176519620-d7c8fb32-a8ba-45bb-b4ad-f5ba51c5e91e.png)

#### Outputs

Here's the Output Predictions


![image](https://user-images.githubusercontent.com/87599801/176519676-d697098a-a583-46b4-96e9-962f5a155a12.png)

And here's the outputted Portfolio Return Probability Distribution


![Returns](https://user-images.githubusercontent.com/87599801/176519788-a8ddbb87-ddd6-4c90-8f6b-a143393a4c0c.png)

The bit filled in red is the area beyond the VaR, our worst-case scenario. Let's hope our portfolio never goes there!


### Using the model

##### Cloning the Repo
First, we have to clone the repo to your local device. Run this code on your terminal
```
git clone https://github.com/ramsundaram101/Monte-Carlo-Portfolio-Predictor
```

##### Installing Dependencies
Our model has the following requirements:
* pandas
* numpy
* datetime
* yfinance
* scipy
* matplotlib
* seaborn
* argparse

To install these dependencies, run
```
pip install -r requirements.txt
```

##### Running the Simulator
We're all set to run the actual simulator now! Just open your Terminal and run 
```
python MonteCarlo.py
```

##### Additional Arguments
Our model has only two additional arguments:
* percent (default = True) - Returns portfolio value in percentage instead of absolute value terms
* compare_market (default = False) - Add True here if you also want to factor overall market performance into your simulations
* alpha (default = 95) - The confidence interval upto which you want to predict VaR and cVaR. Usually we precict with 95% percent confidence

For example, a script with these additional arguments would look like. Hpwever, feel free to omit these arguments and use the defaults.
```
python MonteCarlo.py --percent False --compare_market True --alpha 99
```
