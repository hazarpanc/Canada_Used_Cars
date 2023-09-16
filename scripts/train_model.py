import pandas as pd
import datetime
import plotly.express as px
import category_encoders as ce
import xgboost as xgb



def fit_and_transform_data(X_train, y_train, targetenc_cols, onehotenc_cols):
    """
    Fits TargetEncoder and OneHotEncoder to the training data and transforms the data.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series or np.array): Training target.
    - targetenc_cols (list): List of columns to be target encoded.
    - onehotenc_cols (list): List of columns to be one-hot encoded.

    Returns:
    - pd.DataFrame: Encoded training features encoder.

    """

    # Target encoding
    target_encoder = ce.TargetEncoder(cols=targetenc_cols)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    print(f"Target encoded the columns: {targetenc_cols}")

    # One-hot encoding
    onehot_encoder = ce.OneHotEncoder(cols=onehotenc_cols, use_cat_names=True)
    X_train_encoded = onehot_encoder.fit_transform(X_train_encoded, y_train)
    print(f"One hot encoded the columns: {onehotenc_cols}")

    return X_train_encoded, target_encoder, onehot_encoder



def transform_data(X_test, target_encoder, onehot_encoder):
    """
    Transforms the data using the provided encoders.

    Parameters:
    - X_test (pd.DataFrame): Test features.
    - target_encoder (TargetEncoder): Trained target encoder from fit_and_encode_data.
    - onehot_encoder (OneHotEncoder): Trained one-hot encoder from fit_and_encode_data.

    Returns:
    - pd.DataFrame: Encoded test features.

    """
    # Target encoding
    X_test_encoded = target_encoder.transform(X_test)

    # One-hot encoding
    X_test_encoded = onehot_encoder.transform(X_test_encoded)

    return X_test_encoded


def reverse_one_hot_encoding(df, numerical_column):
    """
    Reverse the one-hot encoding on a Pandas DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing one-hot encoded columns.
    - numerical_column (str): The name of the column with numerical values.

    Returns:
    - pandas.DataFrame: The reversed one-hot encoding as a DataFrame, sorted by the numerical column.
    """
    reversed_encoding = df.drop(numerical_column, axis=1).idxmax(axis=1)
    reversed_encoding = reversed_encoding.str.split('_', expand=True)[1]
    reversed_encoding = pd.concat([reversed_encoding, df[numerical_column]], axis=1)
    reversed_encoding.sort_values(by=numerical_column, inplace=True)

    return reversed_encoding



def calculate_mape_by_make(model, X_data, y_data, make_columns):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) for the model's predictions by each car make.

    Parameters:
    - model: The trained XGBoost model.
    - X_data (pandas.DataFrame): The feature data used for prediction.
    - y_data (pandas.Series): The actual target values.
    - make_columns (list): The list of columns containing one-hot encoded car make information.

    Returns:
    - pandas.DataFrame: The MAPE values by car make.
    """
    # Create DMatrix
    dtrain = xgb.DMatrix(X_data, label=y_data)
    
    # Make predictions on the data
    predictions = model.predict(dtrain)

    # Combine predictions, actual values, and car make information into a DataFrame
    results = pd.DataFrame({'Prediction': predictions, 'Actual': y_data})
    results = pd.concat([results, X_data[make_columns]], axis=1)

    # Calculate absolute percentage error
    results['MAPE'] = abs((results['Actual'] - results['Prediction']) / results['Actual']) * 100

    # Calculate MAPE by car make
    mape_by_make = results.groupby(make_columns)['MAPE'].mean().reset_index()
    
    # Reverse one-hot encoding
    mape_by_make = reverse_one_hot_encoding(mape_by_make, 'MAPE')
    
    # Sort by MAPE values
    mape_by_make = mape_by_make.sort_values("MAPE")

    return mape_by_make



def split_data_by_date(df, date_column, target_column, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split the data into training, validation, and test sets based on a cutoff date.
    If test size is set to 0, it will only create training and validation sets.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.
        date_column (str): The name of the date column in the DataFrame.
        target_column (str): The name of the target column in the DataFrame.
        val_size (float): The proportion of the data to include in the validation set.
        test_size (float): The proportion of the data to include in the test set.
        random_state (int): The random seed for reproducibility.

    Returns:
        X_train (pandas.DataFrame): The features of the training set.
        X_val (pandas.DataFrame): The features of the validation set.
        X_test (pandas.DataFrame or None): The features of the test set.
        y_train (pandas.Series): The target values of the training set.
        y_val (pandas.Series): The target values of the validation set.
        y_test (pandas.Series or None): The target values of the test set.
    """

    # Sort the data by date in ascending order
    df_sorted = df.sort_values(date_column)

    # Calculate the cutoff indices
    val_cutoff = int((1 - val_size - test_size) * len(df_sorted))
    test_cutoff = int((1 - test_size) * len(df_sorted))

    # Split the data into training, validation, and test sets
    train_data = df_sorted.iloc[:val_cutoff, :]
    val_data = df_sorted.iloc[val_cutoff:test_cutoff, :]
    
    X_train = train_data.drop([date_column, target_column], axis=1)
    y_train = train_data[target_column]
    
    X_val = val_data.drop([date_column, target_column], axis=1)
    y_val = val_data[target_column]
    
    # If test_size is set to 0, don't create test set
    if test_size == 0:
        X_test, y_test = None, None
        print("Validation set cutoff date:", val_data[date_column].min(), "- Rows:", len(val_data))
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # If test_size is not 0, create test set as usual
    test_data = df_sorted.iloc[test_cutoff:, :]
    X_test = test_data.drop([date_column, target_column], axis=1)
    y_test = test_data[target_column]
    
    print("Validation set cutoff date:", val_data[date_column].min(), "- Rows:", len(val_data))
    print("Test set cutoff date:", test_data[date_column].min(), "- Rows:", len(test_data))
    return X_train, X_val, X_test, y_train, y_val, y_test



def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the mean absolute percentage error (MAPE) between the true values and the predicted values.
    It first calculates the absolute percentage error for each data point.
    Then, it calculates the mean of the absolute percentage errors to get the MAPE.
    
    Parameters:
        y_true (array-like): The true (actual) values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The mean absolute percentage error (MAPE).

    """
    # Convert both of the variables to arrays
    y_true = pd.Series(y_true).values
    y_pred = pd.Series(y_pred).values
    
    # Check if the input arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")

    # Calculate the absolute percentage error for each data point
    absolute_percentage_errors = [abs((y_true[i] - y_pred[i]) / y_true[i]) for i in range(len(y_true))]

    # Calculate the mean absolute percentage error (MAPE)
    mape = sum(absolute_percentage_errors) / len(y_true)

    return mape


def evaluate_model(model, X_test_encoded, y_test, dmatrix_conversion=False):
    """
    Evaluate a model on the provided test data using MAPE.

    Parameters:
    - model: Trained model.
    - X_test_encoded (pd.DataFrame): Encoded test features.
    - y_test (pd.Series or np.array): Test target.

    Returns:
    - float: MAPE (Mean Absolute Percentage Error) score.

    """
    
    if dmatrix_conversion:
        # Convert the test data into DMatrix format for XGBoost
        X_test_encoded = xgb.DMatrix(X_test_encoded, label=y_test)

    # Make predictions on the test set
    y_pred = model.predict(X_test_encoded)

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return mape



def prepare_data(df, target_col='price', date_col='fetchdate', onehotenc_cols=None, targetenc_cols=None):
    """
    Prepare the data for XGBoost training by separating features and target and applying encoding.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the features and target.
    - target_col (str): The name of the target column. Default is 'price'.
    - date_col (str): The name of the date column to be excluded. Default is 'fetchdate'.
    - onehot_cols (list): List of column names to be one-hot encoded. Default is None.
    - targetenc_cols (list): List of column names to be target encoded. Default is None.

    Returns:
    - pd.DataFrame: Encoded features.
    - pd.Series: Target variable.

    """
    # Separate the features and target
    X = df.drop([target_col, date_col], axis=1)
    y = df[target_col]

    # Encode categorical columns
    X_encoded, target_encoder, onehot_encoder = fit_and_transform_data(X, y, targetenc_cols, onehotenc_cols)

    return X_encoded, y, target_encoder, onehot_encoder