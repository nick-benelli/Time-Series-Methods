import pandas as pd

def aggregate_weekly_data(data, metric_name, datetime_name = 'date', method='sum') -> pd.DataFrame:
    '''
    Takes a time series daily values and aggregates them to weekly
    ---
    Input:
        data : pandas.Series or pandas.DataFrame
        metric_name : str
            column name of values measured
        datetime_name : str
            column name of pd.datetime values
        method : str
            sum, mean max, min: method of aggregation
            The default is 'sum'

    Output:
        pd.DataFrame
    '''
    data_weekly = data.copy().reset_index()
    data_weekly[datetime_name] = pd.to_datetime(data_weekly[datetime_name])
    data_weekly[datetime_name] = data_weekly[datetime_name] - pd.to_timedelta(7, unit='d')

    grouper = data_weekly.groupby([ pd.Grouper(key=datetime_name, freq='W-MON')])[metric_name]
    
    try:
        data_weekly = grouper.apply(method).reset_index().sort_values(datetime_name)
    except ValueError:
        data_weekly = grouper.apply(method).reset_index(drop= True).sort_values(datetime_name)

    data_weekly = data_weekly.set_index([datetime_name])
    return pd.DataFrame(data_weekly)





def aggregate_weekly_data_use_sum(data, metric_name, datetime_name = 'date', use_sum=True) -> pd.Series:
    '''
    Takes a time series daily values and aggregates them to weekly
    ---
    Input:
        data : pandas.Series or pandas.DataFrame
        metric_name : str
            column name of values measured
        datetime_name : str
            column name of pd.datetime values
        use_sum : bool
            True : sum values per week
            False : average values per week
    '''
    if use_sum:
        method = 'sum'
    else:
        method = 'mean'

    return aggregate_weekly_data(data, metric_name, datetime_name, method)
