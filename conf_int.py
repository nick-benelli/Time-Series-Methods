import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_conf_int(actual, predicted, confidence_level = 0.95) -> pd.DataFrame:
    # Assuming predictions and actual values are numpy arrays
    actual = pd.Series(actual)
    predicted = pd.Series(predicted)
    prediction_values = predicted.values
    actual_values = actual.values

    # Calculate residuals
    residuals = actual_values - prediction_values

    # Calculate standard deviation of residuals
    std_dev = np.std(residuals)

    # Set confidence level (e.g., 95%)
    #confidence_level = 0.95

    # Calculate z-score for the desired confidence level
    z_score = np.percentile(residuals, (1 + confidence_level) / 2 * 100)
    z_score = norm.ppf((1 + confidence_level) / 2)

    # Calculate upper and lower bounds for the confidence interval
    upper_bound = prediction_values + z_score * std_dev
    lower_bound = prediction_values - z_score * std_dev
    ci = pd.concat(
        [
            pd.Series(lower_bound, name= 'lower'), 
            pd.Series(upper_bound, name= 'upper')
        ], axis=1)

    try:
        # add index and column names and index to data frame 
        ci.index = actual.index
        if actual.name is None:
            pass
        else:
            ci = ci.add_suffix(f" {actual.name}")
    except AttributeError:
        pass
    ci
    return ci