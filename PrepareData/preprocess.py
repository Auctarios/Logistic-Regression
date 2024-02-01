import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def prepare_data(data, sc = None):
    """
    Prepares the data for model with respect to "Binary Classification with a Bank Churn Dataset" on kaggle.

    Give scaler if the data is test set, else it should be train set.

    Returns:
    ids (array): ID of each data. (can be used to send predictions at the end)
    data_scaled (pd.DataFrame): The data that is scaled.
    scaler (MinMaxScaler): The scaler that is used. (can be used to scale test data after fitting it to train data)
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas.DataFrame object.")
    if sc is not None and not isinstance(sc, MinMaxScaler):
        raise TypeError("Scaler must be a MinMaxScaler")
    

    ids = data['id']
    data.drop(columns=['id'], inplace=True)

    if data.isna().sum().sum() > 0:
        warnings.warn(f"There were missing values, these are deleted.")
        data = data.dropna()

    data['IsSenior'] = data['Age'].apply(lambda x: 1 if x >= 60 else 0)
    data['IsActive_by_CreditCard'] = data['HasCrCard'] * data['IsActiveMember']
    data['Products_Per_Tenure'] = data['Tenure'] / data['NumOfProducts']

    data['male'] = data['Gender'].apply(lambda x: 1 if x.lower() == 'male' else 0)
    data['female'] = data['Gender'].apply(lambda x: 1 if x.lower() == 'female' else 0)
    data['France'] = data['Geography'].apply(lambda x: 1 if x == 'France' else 0)
    data['Spain'] = data['Geography'].apply(lambda x: 1 if x == 'Spain' else 0)
    data['Germany'] = data['Geography'].apply(lambda x: 1 if x == 'Germany' else 0)

    data.drop(columns=['Gender', 'Surname', 'Geography'], inplace=True)

    if sc is not None:
       data_scaled = sc.transform(data)
    else:
        exited = data['Exited']
        data.drop(columns=['Exited'], inplace= True)
        cols = data.columns
        sc = MinMaxScaler()
        data_scaled_ = sc.fit_transform(data)
        data_scaled = pd.DataFrame(data = data_scaled_, columns = cols)
        data_scaled['Exited'] = exited

    return [ids, data_scaled, sc] 