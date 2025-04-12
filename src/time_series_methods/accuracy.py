import pprint
import numpy as np
from sklearn.metrics import r2_score


def find_prediction_acc(y_pred, y_true, print_result=True):
    try:
        y_pred = y_pred.values
        y_true = y_true.values
    except AttributeError:
        pass

    y_pred = y_pred.flatten().reshape(-1, 1)
    y_true = y_true.flatten().reshape(-1, 1)
    mape = np.mean(
        np.abs(y_pred - y_true) / np.abs(y_true)
    )  # Mean absolute percentage error
    mae = np.mean(np.abs(y_pred - y_true))  # mean absolute error
    mpe = np.mean((y_pred - y_true) / y_true)  # Mean percentage error
    rmse = np.mean((y_pred - y_true) ** 2) ** (1 / 2)  # RMSE
    corr = np.corrcoef(y_pred.flatten(), y_true.flatten())[
        0, 1
    ]  # Correlation Coefficient
    r_squared = r2_score(y_true, y_pred)
    # my_r2 = calc_r_squared(y_true, y_pred)

    mins = np.amin(np.hstack([y_pred.reshape(-1, 1), y_true.reshape(-1, 1)]), axis=1)
    maxs = np.amax(np.hstack([y_pred.reshape(-1, 1), y_true.reshape(-1, 1)]), axis=1)
    minmax = 1 - np.mean(mins.reshape(-1, 1) / maxs.reshape(-1, 1))  # minmax

    accuracy_results = {
        "mape": round(mape, 3),
        "mae": round(mae, 3),
        "mpe": round(mpe, 3),
        "rmse": round(rmse, 3),
        "corr": round(corr, 3),
        "minmax": round(minmax, 3),
        "r_square": round(r_squared, 3),
        #'R2_corr' : round(my_r2, 3)
    }

    if print_result:
        pprint.pprint(accuracy_results)
    return accuracy_results


def calc_r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y - y_bar) ** 2).sum()
    ss_res = ((y - y_hat) ** 2).sum()
    return 1 - (ss_res / ss_tot)
