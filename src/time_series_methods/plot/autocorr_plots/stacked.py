# Package
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def plot_acf_pacf_stacked(
    time_series, lags=None, title=None, figsize=(8, 10), color=None, line_color=None
):
    plt.figure(figsize=figsize)
    plt.suptitle(title)
    plt.subplot(211)
    plot_acf(
        time_series,
        ax=plt.gca(),
        lags=lags,
        color=color,
        vlines_kwargs={"colors": line_color},
    )
    plt.subplot(212)
    plot_pacf(
        time_series,
        ax=plt.gca(),
        lags=lags,
        color=color,
        vlines_kwargs={"colors": line_color},
    )
    plt.show()
    return None
