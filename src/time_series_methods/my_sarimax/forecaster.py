def forecast_model(model, steps=None, exog=None):
    """
    Forecast using a fitted SARIMAX model.

    Parameters:
    - model: Fitted SARIMAX model.
    - steps: Number of steps to forecast.
    - exog: exogenous data.

    Returns:
    - fcast = model.get_forecast()
    """
    # Forecast
    fcast = model.get_forecast(steps=steps, exog=exog, print_result=False)
    data_pred = fcast.predicted_mean
    ci = fcast.conf_int()
    ci.index = fcast.predicted_mean.index
    fcast.ci = ci
    fcast.pred_data = data_pred
    fcast.pred_vals = data_pred.values
    return fcast
