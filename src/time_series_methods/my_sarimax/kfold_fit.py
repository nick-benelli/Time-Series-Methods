import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit

from .evaluator import SARIMAXEvaluator
from .forecaster import forecast_model
from ..k_folds_split import k_folds_split


def fit_sarimax_k_fold(time_series, my_order, my_seasonal_order, n_splits=3, exog=None):

    train_idx_list, val_idx_list = k_folds_split(time_series, n_splits)

    df_X = pd.DataFrame(time_series)
    # k = 0

    current_model_names = []
    current_model_results = []
    result_list = []

    for k in range(n_splits):
        print("----------------------------------------------")
        print(f"k={k}")

        # train, val data
        train_idx = train_idx_list[k]
        val_idx = val_idx_list[k]

        data_tr_k, data_val_k = df_X.iloc[train_idx, :], df_X.iloc[val_idx, :]
        y_true = data_val_k.values

        if exog is None:
            use_exog = False
            df_exog_tr_k, df_exog_val_k = (None, None)
        else:
            use_exog = True
            df_exog_tr_k, df_exog_val_k = exog.iloc[train_idx, :], exog.iloc[val_idx, :]

        # Model Fit
        model = SARIMAX(
            data_tr_k,
            exog=df_exog_tr_k,
            order=my_order,
            seasonal_order=my_seasonal_order,
        )
        model_result = model.fit()

        # Forecast
        forecast_steps = len(data_val_k)
        fcast = forecast_model(model_result, forecast_steps, exog=df_exog_val_k)

        # Accuracy Measure
        Evaluator = SARIMAXEvaluator(model_result)
        k_model_name = f"{Evaluator.model_name}-k:{k}exog:{use_exog}"

        acc_dict = Evaluator.train_and_test_accurcy_metrics(data_val_k, df_exog_val_k)
        acc_dict["model_name"] = k_model_name
        acc_dict["k"] = k
        # acc_dict['model_result'] = model_result
        flat_acc_dict = Evaluator.flatten_acc_maps(acc_dict, delimiter=".")

        # Add to lists
        result_list.append(flat_acc_dict)
        # models.append(model)
        current_model_results.append(model_result)
        current_model_names.append(k_model_name)
        return current_model_results, acc_dict
