

def get_model_name(model):
    try:
        model.order
    except:
        model = model.model

    order = model.order
    seasonal_order  = model.seasonal_order

    my_order_str = str(order).replace(' ', '')
    my_seasonal_order_str = str(seasonal_order[0:3]).replace(' ', '')
    season = seasonal_order[3]

    model_name = f"SARIMA{my_order_str}x{my_seasonal_order_str}[{season}]"
    return model_name