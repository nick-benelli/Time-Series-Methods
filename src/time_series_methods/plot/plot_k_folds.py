import matplotlib.pyplot as plt
import pandas as pd


def plot_k_folds_time_series(
    model_results,
    model_names,
    time_series,
    k_fold_tr_idx_list,
    k_fold_val_idx_list,
    metric_um="",
    exog=None,
):

    n_splits = len(k_fold_tr_idx_list)
    # Plot
    line_pred_style = "--"

    plt.figure(figsize=(16, 16))

    for k in range(n_splits):
        # train, val data
        tr_idx = k_fold_tr_idx_list[k]
        val_idx = k_fold_val_idx_list[k]

        # if exog is None;

        data_tr_k, data_val_k = time_series.iloc[tr_idx], time_series.iloc[val_idx]

        if exog is None:
            df_exog_tr_k, df_exog_val_k = (None, None)
        else:
            df_exog_tr_k, df_exog_val_k = exog.iloc[tr_idx, :], exog.iloc[val_idx, :]

        plt.subplot(n_splits, 1, (k + 1))
        plt.plot(
            pd.concat([data_tr_k, data_val_k], axis=0),
            marker="o",
            markersize=3,
            linewidth=2,
            label="true",
        )

        # Model
        model_result = model_results[k]
        model_name = model_names[k]

        fcast = model_result.get_forecast(len(data_val_k), exog=df_exog_val_k)

        plt.plot(
            model_result.fittedvalues,
            marker="o",
            markersize=2,
            linewidth=3,
            label=f"{model_name} fitted",
        )
        plt.plot(
            fcast.predicted_mean,
            line_pred_style,
            markersize=3,
            linewidth=2,
            label=f"{model_name} pred",
        )

        ci = fcast.conf_int()

        plt.fill_between(
            ci.index,
            ci.iloc[:, 0],
            ci.iloc[:, 1],
            alpha=0.1,
            label=f"{model_name} pred ci",
        )
        plt.title(f"{model_name} Val k={k}")
        if (k + 1) == n_splits:
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Time (Weeks)")
        plt.ylabel(metric_um)

    plt.show()
    return None
