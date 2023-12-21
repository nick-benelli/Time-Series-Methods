# Package
import matplotlib.pyplot as plt

# Time Series
import statsmodels.api as sm
"""
__Decomposing using statsmodel:__
- We can use statsmodels to perform a decomposition of this time series. 
- The decomposition of time series is a statistical task that deconstructs a time series into several components, each representing one of the underlying categories of patterns. 
- With statsmodels we will be able to see the trend, seasonal, and residual components of our data.
"""

seasonal_decompose = sm.tsa.seasonal_decompose


def plot_seasonal_decomposition(decomposition, color= None, title=None, figsize=(18,8), markersize=4, fontsize=None):
    """
    __Decomposing using statsmodel:__
        - We can use statsmodels to perform a decomposition of this time series. 
        - The decomposition of time series is a statistical task that deconstructs a time series into several components, each representing one of the underlying categories of patterns. 
        - With statsmodels we will be able to see the trend, seasonal, and residual components of our data.
    """
    
    if title is None:
        title= f"{decomposition.observed.name}: Seasonal Decomposition"
    
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    decomposition.observed.plot(ax=axes[0], legend=False, color=color)
    axes[0].set_ylabel('Observed')
    decomposition.trend.plot(ax=axes[1], legend=False, color=color)
    axes[1].set_ylabel('Trend')
    decomposition.seasonal.plot(ax=axes[2], legend=False, color=color)
    axes[2].set_ylabel('Seasonal')
    decomposition.resid.plot(ax=axes[3], linestyle='', marker='.', markersize=markersize, legend=False, color=color)
    axes[3].set_ylabel('Residual')
    fig.suptitle(title, fontsize=fontsize)
    plt.show()
    return None