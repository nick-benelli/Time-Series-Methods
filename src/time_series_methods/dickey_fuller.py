from statsmodels.tsa.stattools import adfuller

"""
## Stationarity
- A Time Series is said to be stationary if its statistical properties such as mean, variance remain constant over time.
- Most of the Time Series models work on the assumption that the TS is stationary. Major reason for this is that there are many ways in which a series can be non-stationary, but only one way for stationarity.
- Intuitively, we can say that if a Time Series has a particular behaviour over time, there is a very high probability that it will follow the same in the future. 
- Also, the theories related to stationary series are more mature and easier to implement as compared to non-stationary series.
"""


def null_hypothesis_result(p_value):
    if p_value < 0.05:
        print(
            f"The null hypothesis can be rejected. The time-series is stationary. ({round(p_value, 4)} < 0.05)"
        )
    else:
        print(
            f"The null hypothesis cannot be rejected. The time-series is not stationary. ({round(p_value, 4)} > 0.05)"
        )

    return None


def test_time_series(time_series_data):
    dickey_fuller_result = adfuller(time_series_data)
    p_value = dickey_fuller_result[1]
    print(f"Dicker-Fuller p-value: {round(p_value, 4)}")

    # Test Results
    null_hypothesis_result(p_value)

    return dickey_fuller_result
