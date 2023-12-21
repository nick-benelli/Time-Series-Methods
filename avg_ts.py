import pandas as pd


def average_time_series_by_county(df_county, metric_column_name = 'arithmetic_mean'):
    '''
    Average time series data from multiple monitoring sites
    Inputs:
    '''
    site_list = list(df_county['site_id'].unique())

    # Combine it into one for loop
    # Create data frame with dates as index
    dates = list(df_county['datetime'].sort_values().unique())
    #print(dates)
    df_dates = pd.DataFrame(index=dates)

    # Filter by site
    for site_id in site_list:
        #print(site_number)
        #df_site = df_county.loc [ df_county['site_id'] == site_id, ['site_id', 'datetime', 'poc', 'date_of_last_change', metric_column_name]].sort_values(['datetime'])
        df_site = df_county.loc [ df_county['site_id'] == site_id, ['site_id', 'datetime', 'poc', metric_column_name]].sort_values(['datetime'])

        site_id = df_site['site_id'].iloc[0]
        #date_of_last_change = df_site['date_of_last_change'].unique()
        #print(date_of_last_change)
    
        # group by date if different pocs and set index to datetime
        #df_site.index = df_site['datetime']
        df_site = df_site.groupby('datetime').mean(numeric_only = True)

        df_site['site_id'] = site_id

        # add arithemtic mean by site to date df
        site_id = df_site['site_id'].iloc[0]
        df_dates.loc[:, site_id] = df_site.loc[:, metric_column_name]


        # Interpolate arithmetic mean
        max_site_date = df_site.index.max()
        #print(max_site_date)
        df_dates.loc[:max_site_date, :] = df_dates.loc[:max_site_date, :].interpolate()

    county_avg_ts = df_dates.mean(axis=1)
    county_avg_ts.name = metric_column_name
    county_avg_ts.index.name = 'datetime'
    return county_avg_ts