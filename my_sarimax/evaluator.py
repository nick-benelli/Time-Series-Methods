import numpy as np 
import pandas as pd
import flatdict

import accuracy
import my_sarimax.forecaster as forecaster
from . import model_namer


class SARIMAXEvaluator():
    def __init__(self, model_result):
        self.model_result = model_result
        self.model = model_result.model
        self.build_accuracy_metrics = accuracy.find_prediction_acc
        self.model_name = self.get_model_name()
        return None
    

    def get_model_name(self):
        return model_namer.get_model_name(self.model)

    
    def get_fitted_values(self):
        fittvals0 = self.model_result.fittedvalues

        order = self.model_result.model.order
        seasonal_order  = self.model_result.model.seasonal_order
        season = seasonal_order[3]
        I = seasonal_order[1]

        seas_diff_fitted = fittvals0.iloc[(I*season): ]
        diff_fitted = seas_diff_fitted.iloc[order[1]:]
        return diff_fitted
    

    def get_train_data(self):
        original_data_vals = self.model_result.model.endog
        original_index = self.model_result.model.data.row_labels
        columns = [ self.model.endog_names]
        return pd.DataFrame(original_data_vals, index= original_index, columns=columns)
    

    def forecast_model(self, steps =None,exog=None):
        """
        Forecast using a fitted SARIMAX model.

        Parameters:
        - steps: Number of steps to forecast.
        - exog: exogenous validation data

        Returns:
        - fcast = model.get_forecast()
        """
        fcast = forecaster.forecast_model(self.model_result, steps, exog)
        return fcast


    def add_model_acc_metrics(self, acc_dict):
        acc_dict['aic'] = round(self.model_result.aic, 2)
        acc_dict['bic'] = round(self.model_result.bic, 2)
        acc_dict['llf'] = round(self.model_result.llf, 2)
        acc_dict['model'] = self.get_model_name()
        return acc_dict


    def get_model_train_accuracy(self, print_result= False):
        pred_data = self.get_fitted_values()
        true_data = self.get_train_data()
        true_data = true_data.loc[pred_data.index]
        acc_dict = accuracy.find_prediction_acc(pred_data,  true_data, print_result)
        acc_dict = self.add_model_acc_metrics(acc_dict)
        acc_dict['type'] = 'fit'
        return acc_dict


    def get_model_test_accuracy(self, true_data, test_exog_data=None, print_result= False):
        forecas_steps = len(true_data)
        fcast = self.forecast_model(forecas_steps, test_exog_data)
        acc_dict = accuracy.find_prediction_acc(fcast.predicted_mean,  true_data, print_result)
        acc_dict['type'] = 'predict'
        return acc_dict
    

    def train_and_test_accurcy_metrics(self, test_true_data, test_exog_data=None, print_result= False):
        model_accuracy_map = {
            'train' : self.get_model_train_accuracy(print_result),
            'test' : self.get_model_test_accuracy(test_true_data, test_exog_data, print_result)
        }
        return model_accuracy_map


    def flatten_acc_maps(self, model_acc_map, delimiter='.'):
        acc_map_flat = dict(flatdict.FlatDict(model_acc_map, delimiter))
        return acc_map_flat
