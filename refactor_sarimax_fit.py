import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarimax_model(data, exog=None, order=None, seasonal_order=None):
    """
    Fit SARIMAX model.

    Parameters:
    - data: Time series data for training.
    - exog: Exogenous variables.
    - order: SARIMAX order.
    - seasonal_order: SARIMAX seasonal order.

    Returns:
    - Fitted SARIMAX model.
    """
    model = SARIMAX(data, exog=exog, order=order, seasonal_order=seasonal_order).fit()
    return model

def forecast_sarimax_model(model, steps):
    """
    Forecast using a fitted SARIMAX model.

    Parameters:
    - model: Fitted SARIMAX model.
    - steps: Number of steps to forecast.

    Returns:
    - Forecasted values.
    - Confidence interval.
    """
    fcast = model.get_forecast(steps)
    data_pred = fcast.predicted_mean
    ci = fcast.conf_int()
    ci.index = data_pred.index
    return data_pred, ci

def evaluate_sarimax_model(y_true, y_pred, model, k, print_result=False):
    """
    Evaluate SARIMAX model and return accuracy metrics.

    Parameters:
    - y_true: True values.
    - y_pred: Predicted values.
    - model: Fitted SARIMAX model.
    - k: Fold number.
    - print_result: Whether to print the results.

    Returns:
    - Dictionary containing accuracy metrics.
    """
    acc_dict = epa.ts_methods.accuracy.find_prediction_acc(y_pred, y_true, print_result=print_result)
    acc_dict['aic'] = round(model.aic, 2)
    acc_dict['bic'] = round(model.bic, 2)
    acc_dict['llf'] = round(model.llf, 2)
    acc_dict['model'] = f"{model_name}-k:{k}"
    acc_dict['k'] = k
    return acc_dict

def perform_kfold_cross_validation(train_time_series, n_splits, train_idx_list, val_idx_list, use_exogenous,
                                   df_exog_train, my_order, my_seasonal_order, model_name):
    """
    Perform k-fold cross-validation for SARIMAX models.

    Parameters:
    - train_time_series: Time series data.
    - n_splits: Number of splits for k-fold cross-validation.
    - train_idx_list: List of training indices for each fold.
    - val_idx_list: List of validation indices for each fold.
    - use_exogenous: Whether to use exogenous variables.
    - df_exog_train: Exogenous variables for training.
    - my_order: SARIMAX order.
    - my_seasonal_order: SARIMAX seasonal order.
    - model_name: Name of the model.

    Returns:
    - Dictionary containing k-fold cross-validation results.
    """
    k_fold_map = {
        'k': [],
        'train_idx': [],
        'val_idx': [],
        'train_data': [],
        'val_data': [],
        'ytrue': [],
        'model': [],
        'model_name': [],
        'pred_data': [],
        'ypred': [],
        'ci': [],
        'accuracy': [],
    }

    df_X = pd.DataFrame(train_time_series)

    for k in range(n_splits):
        print('----------------------------------------------')
        print(f"k={k}")

        # Extract data for the current fold
        train_idx = train_idx_list[k]
        val_idx = val_idx_list[k]
        data_tr_k, data_val_k = df_X.iloc[train_idx, :], df_X.iloc[val_idx, :]
        y_true = data_val_k.values

        if use_exogenous:
            df_exog_tr_k, df_exog_val_k = df_exog_train.iloc[train_idx, :], df_exog_train.iloc[val_idx, :]
        else:
            df_exog_tr_k, df_exog_val_k = (None, None)

        # Model Fit
        model = fit_sarimax_model(data_tr_k, exog=df_exog_tr_k, order=my_order, seasonal_order=my_seasonal_order)
        k_model_name = f"{model_name}-k:{k}"

        # Forecast
        forecast_steps = len(data_val_k)
        data_pred_k, ci = forecast_sarimax_model(model, forecast_steps)
        y_pred = data_pred_k.values

        # Evaluate
        acc_dict = evaluate_sarimax_model(y_true, y_pred, model, k, print_result=False)

        # Populate k_fold_map dictionary
        k_fold_map['k'].append(k)
        k_fold_map['train_idx'].append(train_idx)
        k_fold_map['val_idx'].append(val_idx)
        k_fold_map['train_data'].append(data_tr_k)
        k_fold_map['val_data'].append(data_val_k)
        k_fold_map['ytrue'].append(y_true)
        k_fold_map['model'].append(model)
        k_fold_map['model_name'].append(k_model_name)
        k_fold_map['pred_data'].append(data_pred_k)
        k_fold_map['ypred'].append(y_pred)
        k_fold_map['ci'].append(ci)
        k_fold_map['accuracy'].append(acc_dict)

    return k_fold_map
