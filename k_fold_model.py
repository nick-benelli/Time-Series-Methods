
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import accuracy

k_fold_map = {
    'k' : [],
    'train_idx' : [],
    'val_idx' : [],
    'train_data' : [],
    'val_data' : [], 
    'ytrue' : [], 

    'model' : [], 
    'model_name' : [], 

    'pred_data' : [],
    'ypred' : [],
    'ci' : [], 
    'accuracy' : [], 
}

def set_train_val_data(train_data, tr_iloc_idxs, val_iloc_idxs):
    data_tr, data_val = train_data.iloc[tr_iloc_idxs], train_data.iloc[val_iloc_idxs]
    return data_tr, data_val


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
    model_result = SARIMAX(data, exog=exog, order=order, seasonal_order=seasonal_order).fit()
    return model_result



    




def     (order, seasonal_order, train_data, k, tr_idxs, val_idxs, df_exog=None):
    
    model_name = get_sarima_model_name(order, seasonal_order)

    # Split training into testing and validation
    tr_data, val_data = set_train_val_data( train_data, tr_idxs, val_idxs)

    if df_exog is not None:
        df_exog_tr, df_exog_val = set_train_val_data( df_exog, tr_idxs, val_idxs)
    else:
         df_exog_tr, df_exog_val = (None, None)


    model = SARIMAX(tr_data, exog=df_exog_tr, order=order, seasonal_order=seasonal_order).fit()
    k_model_name = f"{model_name}-k:{k}"

    # Forecast
    fcast = forecast_model(model, val_data, df_exog_test=df_exog_val)

    # Accuracy Forecast
    acc_dict = accuracy.find_prediction_acc(
        fcast.pred_vals, val_data.values, print_result=False
    )

    acc_dict['aic'] = round(model.aic, 2)
    acc_dict['bic'] = round(model.bic, 2)
    acc_dict['llf'] = round(model.llf, 2)
    acc_dict['model'] = k_model_name
    acc_dict['k'] = k

    k_fold_map['k'].append(k)
    k_fold_map['train_idx'].append(train_idx)
    k_fold_map['val_idx'].append(val_idx)
    k_fold_map['train_data'].append(data_tr_k)
    k_fold_map['val_data'].append(data_val_k)
    k_fold_map['ytrue'].append(y_true)
    # Model
    k_fold_map['model'].append(model)
    k_fold_map['model_name'].append(k_model_name)
    # Predction
    k_fold_map['pred_data'].append(data_pred_k)
    k_fold_map['ypred'].append(y_pred)
    k_fold_map['ci'].append(ci)
    k_fold_map['accuracy'].append(acc_dict)




def build_model_acc_metrics(model, test_data):

    

    

def build_model_accuracy(pred_vals, true_vals):
    accuracy.find_prediction_acc(
        pred_vals, true_vals, print_result=False
    )

    


    acc_dict = epa.ts_methods.accuracy.find_prediction_acc(y_pred, y_true, print_result=False)
    acc_dict['aic'] = round(model.aic, 2)
    acc_dict['bic'] = round(model.bic, 2)
    acc_dict['llf'] = round(model.llf, 2)
    acc_dict['model'] = k_model_name
    acc_dict['k'] = k

    k_fold_map['k'].append(k)
    k_fold_map['train_idx'].append(train_idx)
    k_fold_map['val_idx'].append(val_idx)
    k_fold_map['train_data'].append(data_tr_k)
    k_fold_map['val_data'].append(data_val_k)
    k_fold_map['ytrue'].append(y_true)
    # Model
    k_fold_map['model'].append(model)
    k_fold_map['model_name'].append(k_model_name)
    # Predction
    k_fold_map['pred_data'].append(data_pred_k)
    k_fold_map['ypred'].append(y_pred)
    k_fold_map['ci'].append(ci)
    k_fold_map['accuracy'].append(acc_dict)







def fit_model_k_fold():
    for k in range(n_splits):
        print('----------------------------------------------')
        print(f"k={k}")



