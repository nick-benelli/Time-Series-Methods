def forecast_model(model, steps =None,df_exog_test=None):
    """
    Forecast using a fitted SARIMAX model.

    Parameters:
    - model: Fitted SARIMAX model.
    - steps: Number of steps to forecast.

    Returns:
    - fcast = model.get_forecast()
    """
    # Forecast
    fcast = model.get_forecast(steps=steps, exog=df_exog_test,  print_result= False)
    data_pred = fcast.predicted_mean
    ci = fcast.conf_int()
    ci.index = fcast.predicted_mean.index
    fcast.ci = ci
    fcast.pred_data = data_pred
    fcast.pred_vals = data_pred.values
    return fcast