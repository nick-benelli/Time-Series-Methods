

# Package
import matplotlib.pyplot as plt

# Time Series
import statsmodels.api as sm


def plot_acf_pacf_side_by_side(time_series, lags = None, figsize = (12,4), color= None, title=None):
    # acf pacf plts
    fig, ax = plt.subplots(1,2,figsize= figsize)
    fig.suptitle(title)
    # acf
    sm.graphics.tsa.plot_acf(time_series.values.squeeze(), lags=lags, ax=ax[0], color= color)
    #sm.graphics.tsa.plot_acf(data.values.squeeze(), ax=ax[0], color=color_plot)
    # pacf
    sm.graphics.tsa.plot_pacf(time_series.values.squeeze(), lags=lags, ax=ax[1], color=color)
    plt.tight_layout()
    plt.show()
