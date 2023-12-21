

from statsmodels.tsa.stattools import adfuller


def null_hypothesis_result(p_value):
    if p_value < 0.05:
        print(f'The null hypothesis can be rejected. The time-series is stationary. ({round(p_value, 4)} < 0.05)')
    else:
        print(f'The null hypothesis cannot be rejected. The time-series is not stationary. ({round(p_value, 4)} > 0.05)')
    
    return None


def test_time_series(time_series_data):
    dickey_fuller_result = adfuller(time_series_data)
    p_value = dickey_fuller_result[1]
    print(f'Dicker-Fuller p-value: {round(p_value, 4)}')

    # Test Results
    null_hypothesis_result(p_value)

    return dickey_fuller_result

