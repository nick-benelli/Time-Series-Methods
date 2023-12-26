import pandas as pd
from IPython.display import Image, display


def count_data_points_by_year(time_series):
    df_counts_by_year = pd.DataFrame(time_series.index.year.value_counts().sort_index())
    df_counts_by_year.columns = ['counts']
    display(df_counts_by_year)

    print("Median Count", df_counts_by_year.median())
    display(df_counts_by_year.describe())


    period_suggested = int(df_counts_by_year['counts'].median())
    print(period_suggested)
    return df_counts_by_year