import seaborn as sns
import matplotlib.pyplot as plt

def plot(time_series, figsize=(15,6), color=None, title='Yearly Box and Whisker Plots'):
    '''
    __Box and Whisker Plots:__
        - Median values across years confirms an upwards trend
        - Steady increase in the spread, or middle 50% of the data (boxes) over time
        - A model considering seasonality might work well
    '''
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        x = time_series.index.year, 
        y= time_series.values, 
        ax=ax, color=color
    )
    plt.title(title)
    plt.show()
    return None