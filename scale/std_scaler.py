from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_train_test(scaler, train_data, test_data):
    scaler = scaler.fit(train_data)
    train_data_scl_array = scaler.transform(train_data)
    test_data_scl_array = scaler.transform(test_data)

    train_data_scl = train_data.copy()
    train_data_scl.iloc[:, :] = train_data_scl_array

    test_data_scl = test_data.copy()
    test_data_scl.iloc[:, :] = test_data_scl_array
    return scaler, train_data_scl, test_data_scl

