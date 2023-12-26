import matplotlib.pyplot as plt
import pandas as pd
"""
### Plotting Rolling Statistics
- We observe that the rolling mean and Standard deviation are not constant with respect to time (increasing trend)
- The time series is hence not stationary
"""

def calc_and_plot_rolling_mean_and_std(time_series, window=4, show_plot=True, color=None, linewidth=None, title=None) -> (pd.Series, pd.Series):
    """
    ### Plotting Rolling Statistics
    - We observe that the rolling mean and Standard deviation are not constant with respect to time (increasing trend)
    - The time series is hence not stationary
    """
    #Determing rolling statistics
    rolmean = time_series.rolling(window).mean()
    rolstd = time_series.rolling(window).std()

    #Plot rolling statistics:
    if show_plot:
        plt.figure(figsize=(9,4))
        orig = plt.plot(time_series,label='Original', linewidth = linewidth, color=color)
        mean = plt.plot(rolmean, label='Rolling Mean',  linewidth = linewidth, color='r', linestyle = '--')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std',  linewidth = linewidth, linestyle= '-')
        plt.legend(loc='best')
        if title is None:
            plt.title(f'Rolling Mean & Standard Deviation Window={window}')
        else:
            plt.title(title)
        plt.show(block=False)

    return rolmean, rolstd