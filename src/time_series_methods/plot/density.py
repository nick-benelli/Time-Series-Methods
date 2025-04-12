# Package
import matplotlib.pyplot as plt


def plot_density_distibution(time_series, kind="kde", color=None, title="Distribution"):
    """
    __Reviewing plots of the density of observations can provide further insight into the structure of the data:__
        - The distribution is not perfectly Gaussian (normal distribution).
        - The distribution is left shifted.
        - Transformations might be useful prior to modelling.

    """
    plt.figure(1)
    plt.suptitle(title)
    # plt.suptitle('Interpolated Daily Distribution')
    plt.subplot(211)
    time_series.hist(color=color)
    plt.subplot(212)
    time_series.plot(kind="kde", color=color)
    plt.show()
    return None
