import pandas as pd
import numpy as np
import os
import datetime
import warnings
import logging
import boto3
from trim_cleaning_helper import *
from sagemaker import get_execution_role
warnings.filterwarnings("ignore")

logging.basicConfig(
    filename='./logs/preprocessing_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def download_file_from_s3(bucket_name, file_key, destination_file):
    """
    Download a file from an S3 bucket.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - file_key (str): The key (path) of the file in the S3 bucket.
    - destination_file (str): The local path where the file will be saved.

    Returns:
    - bool: True if the file was successfully downloaded, False otherwise.
    """

    try:
        # Connect to S3 bucket
        s3 = boto3.resource('s3')
        
        # Download file
        s3.Bucket(bucket_name).download_file(file_key, destination_file)
        print("File downloaded successfully.")
        logging.info(f"SUCCESS: File downloaded successfully.")
        
        return True

    except Exception as e:
        logging.error("ERROR: File could not be downloaded from S3")
        print("Error occurred:", str(e))
        return False
    
    
    
def read_csv_file(file_path, index_col=None):
    """
    Read a CSV file and return a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.
    - index_col (int or str, optional): Column to use as the DataFrame index.

    Returns:
    - pandas.DataFrame: The DataFrame containing the CSV data.
    """

    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col=index_col)
        return df

    except Exception as e:
        logging.error("ERROR: CSV file could not be read")
        print("Error occurred:", str(e))
        return None

    # Check to make sure that the df has data
    try:
        assert df.shape[0] > 0
        logging.info(f"SUCCESS: File read successfully.")
    except:
        logging.error("ERROR: The df has 0 rows")
        
        

def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns from the provided DataFrame.

    This function removes the columns that are deemed unnecessary for further analysis or processing.
    The specified columns are dropped in-place from the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame from which to drop the columns.

    Returns:
    - None: The function modifies the DataFrame in-place.

    """
    
    # Add column for fueltype
    df['fueltype'] = "gas"
    
    # Rename splashBodyType to bodytype
    df = df.rename({'splashBodyType': 'bodytype'}, axis=1)
    
    # Specify which columns to drop
    columns_to_drop = [
        'adId', 'category', 'dealerCoForeignId', 'dealerCoId', 'dealerCoName',
        'foreignId', 'isAmvic', 'isCommercial', 'isMobielRequest', 'strikeThroughPrice',
        'isNpv', 'isNew', 'sourceId', 'pathBodyType', 'status', 'stockNumber',
        'vin', 'odometerCondition', 'mileage', 'isPrivate', 'microSite', 'showSplashPlus',
        'priceAnalysis', 'vehicleAge', 'location', 'carfax', 'items', 'options',
        'priceAnalysisDescription', 'description', 'ssoUserInfo'
    ]

    # Drop the specified columns from the DataFrame
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    # Test to make sure that we have all the expected columns in data
    try:
        assert df.shape[1] > 0
        logging.info(f"SUCCESS: Columns dropped successfully")
    except:
        logging.error("ERROR: The df doesn't have all of the expected columns")
    
    return df



def drop_unnecessary_rows(df):
    """
    Drop rows that are outliers from the provided DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame from which to drop the columns.

    Returns:
    - None: The function modifies the DataFrame in-place.

    """
    
    # List of values to remove
    dealers_to_remove = ["First Choice Auto Salvage","VAUGHAN FINE TOUCH AUTO COLLISION INC.","M.E.M AUTO CLINIC INC.","LUCKYDOG MOTORS"]
    df = df.drop(df[df['dealerCoName'].isin(dealers_to_remove)].index)
    
    # Rebuilt damaged and salvage vehicles
    df['description'].fillna('no description', inplace=True)
    df['description'] = df['description'].str.lower()
    
    # Words to search for in the description column
    words_to_remove = ["rebuilt", "reconstruit", "salvage", "récupération", "recuperation", "damaged", "véhicule accidenté", "transmission needs repair","may need repair"]

    # Combine the words into a regular expression pattern
    pattern = '|'.join(words_to_remove)

    # Use str.contains() to filter out rows
    df = df[~df['description'].str.contains(pattern, case=True, na=False)]
    
    return df


def process_make(df):
    """
    Process the 'make' column of the provided DataFrame.

    This function filters out rows that do not have a 'make' value in the list of valid makes.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'make' column to be processed.

    Returns:
    - pandas.DataFrame: The DataFrame with the specified rows filtered out.

    """
    # List of valid makes
    valid_makes = ["Audi", "BMW", "Mercedes-Benz", "Cadillac", "Chevrolet", "Ford", "Chrysler", "Dodge", "Fiat", "GMC",
                   "Honda", "Hyundai", "Infiniti", "Jaguar", "Jeep", "Kia", "Land Rover", "Lexus", "Lincoln", "Mazda",
                   "Ram", "MINI", "Mitsubishi", "Nissan", "Porsche", "Subaru", "Tesla", "Toyota", "Volkswagen", "Volvo"]

    # Filter out rows that do not have a 'make' value in the list of valid makes
    df = df[df["make"].isin(valid_makes)]
    
    # Convert 'make' column values to lowercase
    df['make'] = df['make'].str.lower()
    
    logging.info(f"SUCCESS: Make column processed successfully")

    return df


def remove_rare_models(df, min_occurrences=5):
    """
    Remove rows with model values that occur fewer than a specified number of times in the DataFrame.

    """
    # Count the occurrences of each model value
    model_counts = df['model'].value_counts()

    # Get the model values that occur fewer than min_occurrences times
    rare_models = model_counts[model_counts < min_occurrences].index

    # Remove rows with rare model values
    filtered_df = df[~df['model'].isin(rare_models)]

    return filtered_df
    
    
    
def process_model(df, min_occurrences=5):
    """
    Process the 'model' column of the provided DataFrame.

    This function cleans and standardizes the 'model' values by converting them to lowercase.
    It also applies specific mappings to fix variations and manually correct model names.
    Additionally, it removes models that occur below a certain threshold to avoid overfitting.
    'Other/Unspecified' models and missing values are replaced with NaN.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'model' column to be processed.

    Returns:
    - pandas.DataFrame: The DataFrame with the 'model' values cleaned, standardized, and combined with the 'make' column.

    """
    # Convert 'model' column values to lowercase
    df['model'] = df['model'].str.lower()

    # STEP 1: Fix French variations of model names
    # Dictionary mapping French variations to English translations
    model_translation = {
        "hybride rechargeable": "plug-in hybrid",
        "portes": "door",
        "hybride": "hybrid",
        "hayon": "hatchback",
        "berline": "sedan",
        "coupé" : "coupe",
        "portes": "door",
        "décapotable" : "convertible",
    }

    # Replace model names using the dictionary
    df['model'] = df['model'].replace(model_translation, regex=True)
    
    
    # STEP 2: Update trim based on model
    # Example mapping: For 'ModelA', change trim to 'TrimX' and model to 'NewModelA'
    mapping = {
        "128i": {"trim": "128i", "model": "1 series"},
        "230": {"trim": "230i", "model": "2 series"},
        "230i xdrive": {"trim": "230i xdrive", "model": "2 series"},
        "228i": {"trim": "228i", "model": "2 series"},
        "330i": {"trim": "330i", "model": "3 series"},
        "340": {"trim": "340i", "model": "3 series"},
        "340i xdrive": {"trim": "340i xdrive", "model": "3 series"},
        "320i": {"trim": "320i", "model": "3 series"},
        "328": {"trim": "328i", "model": "3 series"},
        "328i": {"trim": "328i", "model": "3 series"},
        "328i xdrive": {"trim": "328i xdrive", "model": "3 series"},
        "328d": {"trim": "328d", "model": "3 series"},
        "335i": {"trim": "335i", "model": "3 series"},
        "330i xdrive": {"trim": "330i xdrive", "model": "3 series"},
        "428i": {"trim": "428i", "model": "4 series"},
        "440": {"trim": "440i", "model": "4 series"},
        "440 gran coupe": {"trim": "440i gran coupe", "model": "4 series"},
        "435i": {"trim": "435i", "model": "4 series"},
        "430i xdrive": {"trim": "430i xdrive", "model": "4 series"},
        "435i xdrive": {"trim": "435i xdrive", "model": "4 series"},
        "440i xdrive": {"trim": "440i xdrive", "model": "4 series"},
        "528i": {"trim": "528i", "model": "5 series"},
        "540": {"trim": "540i", "model": "5 series"},
        "530e": {"trim": "530e", "model": "5 series"},
        "528i xdrive": {"trim": "528i xdrive", "model": "5 series"},
        "530i xdrive": {"trim": "530i xdrive", "model": "5 series"},
        "530": {"trim": "530i", "model": "5 series"},
        "530i": {"trim": "530i", "model": "5 series"},
        "540i": {"trim": "540i", "model": "5 series"},
        "750i": {"trim": "750i", "model": "7 series"},
        # Mercedes-Benz
        "glc300": {"trim": "glc300", "model": "glc-class"},
        "c300": {"trim": "c300", "model": "c-class"},
        "gla250": {"trim": "gla250", "model": "gla-class"},
        "cla250": {"trim": "cla250", "model": "cla-class"},
        "gle350": {"trim": "gle350", "model": "gle-class"},
        "a250": {"trim": "a250", "model": "a-class"},
        "a220": {"trim": "a220", "model": "a-class"},
        "gle400": {"trim": "gle400", "model": "gle-class"},
        "gle450": {"trim": "gle450", "model": "gle-class"},
        "gls450": {"trim": "gls450", "model": "gls-class"},
        "e450": {"trim": "e450", "model": "e-class"},
        "b250": {"trim": "b250", "model": "b-class"},
        "e400": {"trim": "e400", "model": "e-class"},
        "s560": {"trim": "s560", "model": "s-class"},
        "e300": {"trim": "e300", "model": "e-class"},
        "glb250": {"trim": "glb250", "model": "glb-class"},
        "e350": {"trim": "e350", "model": "e-class"},
        "glk250": {"trim": "glk250", "model": "glk-class"},
        "glk350": {"trim": "glk350", "model": "glk-class"},
        "ml350": {"trim": "ml350", "model": "ml-class"},
        "s580": {"trim": "s580", "model": "s-class"},
        "s550": {"trim": "s550", "model": "s-class"},
        "g550": {"trim": "g550", "model": "g-class"},
        "gls580": {"trim": "gls580", "model": "gls-class"},
        "cls450": {"trim": "cls450", "model": "cls-class"},
        "c300 4matic": {"trim": "c300 4matic", "model": "c-class"},
        # Tesla
        'model 3 standard plus': {"trim": "standard range plus", "model": "model 3"},
        'model 3 long range': {"trim": "long range", "model": "model 3"},
        'model y long range': {"trim": "long range", "model": "model y"},
        'model x long range': {"trim": "long range", "model": "model x"}
    }

    def update_trim_based_on_model(row, mapping):
        """
        Updates the 'trim' and 'model' values based on the 'model' value using a provided mapping, 
        while preserving other columns in the dataframe.

        Parameters:
        - row (Series): A row from the dataframe with 'model' and 'trim' columns.
        - mapping (dict): A dictionary that maps existing model names to new 'model' and 'trim' values.

        Returns:
        - Series: A series with updated 'model' and 'trim' values along with unchanged values from other columns.
        """

        # Check if the model of the row exists in the mapping
        if row['model'] in mapping:
            # Cache the model's mapping to avoid the KeyError
            model_mapping = mapping[row['model']]

            # Now update the row using the cached mapping
            row['model'] = model_mapping['model']
            row['trim'] = model_mapping['trim']

        # Return the row, whether modified or original
        return row

    # Apply the function to each row, passing the mapping and update in place
    df = df.apply(lambda row: update_trim_based_on_model(row, mapping), axis=1)


    # STEP 3: Standardize model names according to mappings
    # Nested mappings for each make
    car_mappings = {
        'bmw': 
            {
            "2-series": "2 series",
            "4-series": "4 series",
            "4-series": "4 series",
            "2-series": "4 series",
            },
        
        'mercedes-benz': 
            {
            'gla': 'gla-class',
            'glb': 'glb-class',
            'glc': 'glc-class',
            'gle': 'gle-class',
            'gls': 'gls-class',
            'cla': 'cla-class',
            'cls': 'cls-class',
            'amg gle 43':'gle43 amg',
            'amg glc 43':'glc43 amg',
            'amg gla 45':'gla45 amg',
            'amg cla 45':'cla45 amg',
            'amg e 63':'e63 amg',
            'amg c 63':'c63 amg',
            },
        
        "kia": 
            {
            'forte5': 'forte',
            'forte 5-door': 'forte',
            'rio5': 'rio',
            'rio 5-door' : 'rio',
            'niro plug in hybrid' :'niro plug-in hybrid',
            'niro phev' :'niro plug-in hybrid',
            },
        
        "chevrolet":
        
            {
            '2500': 'silverado 2500',
            '1500': 'silverado 1500',
            'silverado': 'silverado 1500',
            '3500' : 'silverado 3500',
            'avalanche 1500': 'avalanche',
            'bolt ev' :'bolt',
            'corvette stingray' : 'corvette',
            },
        
        "chrysler":
            {
            '300c': '300',
            '300s': '300',
            },
        
        "ford":
            {
            'cargo': 'other/unspecified',
            'convertible': 'other/unspecified',
            'fourgon' :'transit cargo van',
            },
        
        "audi":
            {
            'a3 cabriolet': 'a3',
            'a3 sportback': 'a3',
            'a3 berline': 'a3',
            'a4 allroad': 'a4',
            'sedan a4': 'a4',
            'a4 quattro progressiv': 'a4',
            'berline a4': 'a4',
            'a5 sportback': 'a5',
            'a5 cabriolet': 'a5',
            'a5 coupé': 'a5',
            'a5 coupe': 'a5',
            'a6 3.0t quattro': 'a6',
            'a6 allroad': 'a6',
            'a7 sportback': 'a7',
            's3 sedan': 's3',
            's5 sportback': 's5',
            's5 coupe': 's5',
            's5 cabriolet': 's5',
            's6 sedan': 's6',
            'rs 3 sedan': 'rs3',
            'rs 5 sportback': 'rs5',
            'rs 5 coupe': 'rs5',
            'rs 5 coupé': 'rs5',
            'rs 6 avant': 'rs6',
            'rs 7 sportback': 'rs 7',
            'tt coupe': 'tt',
            'tt coupé': 'tt',
            'tts coupé': 'tts',
            'tts coupe': 'tts',
            'tt rs coupe': 'tt rs',
            'r8 coupe': 'r8',
            'r8 coupé': 'r8',
            'q5 sportback': 'q5',
            'q7 technik': 'q7',
            'sq5 sportback': 'sq5',
            },
        
        "dodge":
            {
            'ram' : 'other/unspecified',
            'ram 1500' : 'other/unspecified',
            },
        
        "gmc":
            {
            'sierra': 'sierra 1500',
            '1500': 'sierra 1500',
            '2500': 'sierra 2500',
            '3500': 'sierra 3500',
            '3500': 'sierra 3500',
            'sierra 1500 pickup': 'sierra 1500',
            'new sierra 1500 crew cab 4x4': 'sierra 1500'
            },
    
        "porsche":
            {
            '718 cayman': 'cayman',
            '718 boxster': 'boxster',
            },
    
        "ram":
            {
            'promaster fourgonnette utilitaire' :'promaster cargo van',
            'silverado 1500' :'1500',
            '1500 classic' :'1500',
            'silverado 2500' :'2500',
            'silverado 3500' :'3500',
            '1500 crew cab' :'1500',
            '1500 quad cab' :'1500',
            'promaster city wagon' :'promaster city',
            },
        
        "mini":
            {
            'cooper hardtop' : 'cooper 3 door',
            'cooper 3 door':'cooper 3 door',
            'cooper coupe':'cooper 3 door',
            'hatchback' : 'cooper 3 door',
            'cooper' : 'cooper 3 door',
            '3 door' : 'cooper 3 door',
            '3 portes' : 'cooper 3 door',
            '5 door' : 'cooper 5 door',
            '5 portes' : 'cooper 5 door',
            'cooper convertible' : 'cooper roadster',
            'cabriolet' : 'cooper roadster',
            'convertible' : 'cooper roadster',
            'coupe' : 'cooper coupe',
            'countryman' : 'cooper countryman',
            
            },
    
        "subaru":
            {
            'sti' : 'wrx sti',
            'impreza wrx' : 'wrx',
            'impreza wrx sti' : 'wrx sti'
            },
        
        "tesla":
            {
            'model s standard plus' : 'model s',
            },
        }

                          
    def apply_make_model_mapping(df, mappings):
        """
        Apply make-specific model name mappings to a dataframe.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing columns 'make' and 'model'.
        - mappings (dict): A nested dictionary where the top-level key is the make, 
                           and the corresponding value is a dictionary of model mappings.

        Returns:
        - pd.DataFrame: The dataframe with model names replaced based on the provided mappings.
        """
        # Iterate over each make and its model mapping
        for make, model_mapping in mappings.items():
            # Create a mask for rows that match the current make
            mask = df['make'] == make
            
            # Apply the model mapping only to rows that match the current make
            df.loc[mask, 'model'] = df.loc[mask, 'model'].replace(model_mapping)
            
        return df
    
    # Apply the function
    df = apply_make_model_mapping(df, car_mappings)

    # STEP 4: Remove models that occur below the threshold to avoid overfitting
    df = remove_rare_models(df, min_occurrences=min_occurrences)

    # STEP 5: Remove rows that have 'Other/Unspecified' as their model, and missing values
    df['model'] = df['model'].replace('s?lectionner', np.nan)
    df['model'] = df['model'].replace('other/unspecified', np.nan)
    
    # Remove any rows that have nan as model value
    df.dropna(subset=['model'], inplace=True)
        
    logging.info(f"SUCCESS: Model column processed successfully")

    return df



def process_transmission(df):
    """
    Process the 'transmission' column of the provided DataFrame.

    This function creates a new column 'transmission_manual' indicating whether the transmission is manual or not.
    It then drops the original 'transmission' column from the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'transmission' column to be processed.

    Returns:
    - pandas.DataFrame: The DataFrame with the 'transmission_manual' column added and the 'transmission' column dropped.

    """
    
    # Create a new column indicating manual transmission
    df['transmission_manual'] = df['transmission'].apply(lambda x: 1 if "manual" in str(x).lower() else 0)

    # Drop the original 'transmission' column
    df.drop('transmission', axis=1, inplace=True)
    
    logging.info(f"SUCCESS: Transmission column processed successfully")

    return df



def process_drivetrain(df):
    """
    Process the 'drivetrain' column of the provided DataFrame.

    This function applies transformations to standardize the values in the 'drivetrain' column.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'drivetrain' column to be processed.

    Returns:
    - pandas.DataFrame: The DataFrame with the 'drivetrain' column modified.

    """
    # Drivetrain mapping
    drivetrain_mapping = {"4x4": "AWD", "4X4": "AWD", "4WD": "AWD", "2WD": "FWD", "Not Available": "AWD"}

    # Replace values using the drivetrain mapping
    df['drivetrain'] = df['drivetrain'].replace(drivetrain_mapping)

    # Fill missing values with 'AWD'
    df['drivetrain'].fillna('AWD', inplace=True)
    
    logging.info(f"SUCCESS: Drivetrain column processed successfully")

    return df




def process_bodytype(df):
    """
    Process the 'bodytype' column of the provided DataFrame.

    This function renames the 'splashBodyType' column to 'bodytype' and applies transformations to standardize the body type values.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'splashBodyType' column to be processed.

    Returns:
    - pandas.DataFrame: The DataFrame with the 'bodytype' column modified.

    """
    
    # Standardize bodytype values
    df['bodytype'] = df['bodytype'].apply(lambda x: "truck" if ("truck" in str(x)) else x)
    df['bodytype'] = df['bodytype'].apply(lambda x: "truck" if ("cab " in str(x)) else x)
    df['bodytype'] = df['bodytype'].apply(lambda x: "truck" if (" cab" in str(x)) else x)
    df['bodytype'] = df['bodytype'].apply(lambda x: "truck" if ("super crew" in str(x)) else x)
    df['bodytype'] = df['bodytype'].apply(lambda x: "truck" if ("cutaway" in str(x)) else x)
    df['bodytype'] = df['bodytype'].apply(lambda x: "station wagon" if ("wagon" in str(x)) else x)
    df['bodytype'] = df['bodytype'].apply(lambda x: "minivan" if ("van" in str(x)) else x)
    df['bodytype'] = df['bodytype'].apply(lambda x: "convertible" if ("cabriolet" in str(x)) else x)
    df['bodytype'] = df['bodytype'].apply(lambda x: "convertible" if ("roadster" in str(x)) else x)
    df['bodytype'] = df['bodytype'].apply(lambda x: "hatchback" if ("compact" in str(x)) else x)
    
    logging.info(f"SUCCESS: Body type column processed successfully")
    
    return df




def process_province(df):
    """
    Process the 'url' column of the provided DataFrame to extract and fill missing province information.

    This function extracts the province from the 'url' column and creates a new column 'province' with the extracted values.
    It then drops the original 'url' column from the DataFrame.
    Finally, it fills any missing values in the 'province' column with the default value 'ontario'.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'url' column to be processed.

    Returns:
    - pandas.DataFrame: The DataFrame with the 'province' column added, the 'url' column dropped, and missing values filled.

    """

    def province_from_url(url):
        """Extracts the province from a given URL."""
        main_provinces = ['ontario','quebec']

        # Convert the URL value to a string
        url = str(url) 

        try:
            # Extract the province from the url
            province = url.split('/')[7] 
        except:
            # Default province is set to Ontario if extraction fails
            province = 'ontario' 

        # Check if the extracted province is not in the main provinces
        if province not in main_provinces: 
            province = np.nan # If not, set the province value to NaN (missing value)

        return province
    
    # Apply province extraction function to 'url' column
    df['province'] = df['url'].apply(lambda url: province_from_url(url))

    # Drop the original 'url' column
    df.drop('url', axis=1, inplace=True)

    # Fill missing values in 'province' column with 'ontario'
    df['province'].fillna('ontario', inplace=True)
    
    logging.info(f"SUCCESS: Province column processed successfully")

    return df



def process_odometer(df):
    """
    Process the 'odometer' column of the provided DataFrame.

    This function cleans the 'odometer' values by removing units and commas and converting them to integers.
    It then removes rows that do not fulfill the condition of having more than 1000 kilometers if the car is more than 2 years old.
    Finally, it removes rows where the 'odometer' value exceeds 290,000 kilometers.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'odometer' and 'year' columns to be processed.

    Returns:
    - pandas.DataFrame: The DataFrame with the 'odometer' values cleaned and the specified rows removed.

    """
    # Function to clean the 'odometer' string
    def clean_odometer(value):
        '''Clean and convert the 'odometer' value to an integer.'''
        
        # Cleaning steps to remove units and commas
        value = str(value)
        value = value.replace(" KM", "")
        value = value.replace(",", "")

        try:
            # Convert the cleaned value to an integer
            value = int(value)
        except:
            # Return np.nan if the conversion fails
            value = np.nan

        return value
    
    # Apply clean_odometer function to the odometer column
    df['odometer'] = df['odometer'].apply(clean_odometer)
    
    # Identify rows that don't fulfill the condition of
    # having more than 1000 kilometers if the car is more than 2 years old.
    invalid_rows = (df['year'] < 2021) & (df['odometer'] < 1000)

    # Remove rows that don't fulfill the condition
    df = df[~invalid_rows]

    # Remove rows where 'odometer' value exceeds 290,000 kilometers
    df = df[df['odometer'] <= 290000]
    
    logging.info(f"SUCCESS: Odometer column processed successfully")

    return df



def process_price(df):
    """
    Process the 'price' column of the provided DataFrame.

    This function drops rows with missing 'price' values, removes commas from the 'price' values, 
    and converts them to integers. It also filters out rows where the 'price' value exceeds 250,000
    in an effort to remove outliers.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'price' column to be processed.

    Returns:
    - pandas.DataFrame: The DataFrame with the 'price' values cleaned and the specified rows removed.

    """
    # Drop rows with missing 'price' values, as 'price' is our target variable
    df.dropna(subset=['price'], inplace=True)

    # Cleaning function to remove commas and convert 'price' values to integers
    def clean_price(value):
        '''Remove commas and convert 'price' value to an integer.'''
        return int(str(value).replace(",", ""))

    # Apply the clean_price function to 'price' column
    df['price'] = df['price'].apply(clean_price)

    # Filter out rows where 'price' value is unreasonable
    df = df[df['price'] < 250000]
    df = df[df['price'] > 3000]
    
    logging.info(f"SUCCESS: Price column processed successfully")

    return df



def process_year(df):
    """
    Filter out rows where the car model year exceeds the next year. It ensures that the car model year
    cannot be higher than the next year to maintain data integrity.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the 'year' column to be filtered.

    Returns:
    - pandas.DataFrame: The filtered DataFrame where the car model year is less than or equal to the next year.

    """
    import datetime

    # Get the next year
    next_year = datetime.datetime.now().year + 1

    # Filter the DataFrame based on the car model year
    df = df[df['year'] <= next_year]

    # Feature engineering
    # Calculate time difference from the reference date: 2022-01-01
    df['fetchdate'] = pd.to_datetime(df['fetchdate']) 
    reference_date = pd.to_datetime('2022-01-01')
    df['days_since_reference'] = (df['fetchdate'] - reference_date).dt.days

    # Calculate the age of the car at fetchdate
    df['car_age'] = (df['fetchdate'] - df['year'].apply(lambda x: datetime.datetime(x, 1, 1))) / pd.Timedelta(days=365)
    
    
    #datetime.datetime(year, 1, 1)
    #df['car_age'] = (df['fetchdate'].dt.year - df['year'])
    
    logging.info(f"SUCCESS: Year column processed successfully")
    
    return df



def filter_expensive_trims(data):
    # Exclude data points with expensive trim values
    expensive_trims = ['m340i', 'x3-m40i', 
                       'camaro-2ss', 'camaro-zl1', 'camaro-ss', 'corvette-z06', 
                       'durango-srt', 'charger-srt', 'charger-rt', 'charger-scat', 'challenger-srt', 'challenger-scat', 
                       'f-150-raptor', 'focus-rs', 'mustang-shelby', 'f-150-limited', 'f-150-platinum', 
                       'sierra 1500-denali', 'q50-red', 
                       'grand cherokee-srt', 'wrangler-unlimited', 'wrangler-rubicon', 
                       'range rover sport-svr', 'range rover sport-v8', 'range rover sport-autobiography', 'range rover evoque-hse','range rover-autobiography',
                       'cx-3-gt', 
                       'a-class-amg','s-class-amg',
                       'c-class-c43', 'c-class-c63', 'c-class-c63s', 'c-class-amg', 'cla-class-amg', 'e-class-amg', 'gla-class-amg', 'glc-class-amg', 
                       'gle-class-amg', 'gle-class-gle43', 's-class-maybach', 's-class-s63', 'c-class-amg43', 'glc-class-43', 'glc-class-glc43', 'gle-class-63', 'gle-class-63s', 
                       'gle-class-43', 'cla-class-cla45', 'gla-class-gla45', 
                       'cooper 3 door-john cooper works','cooper clubman-john cooper works','cooper countryman-john cooper works','cooper roadster-john cooper works', 
                       # Old
                       'silverado 1500-trx', 
                       'impreza-wrx', 
                       'rav4-hybrid', 
                       'corolla-hybrid', 
                       'jetta-gli', 
                       'xc60-t8'
                       # Mini
                       'cooper 3 door-se', 'cooper 3 door-john cooper works','cooper roadster', 'cooper countryman-john cooper works', 
                       'cooper clubman-john cooper works', 'cooper paceman-john cooper works', 'cooper s-convertible',                       
                       # Porsche
                       '911-gt3 rs', '911-turbo s convertible', '911-turbo', '911-turbo s','911-gt3','911-targa 4 gts',
                       'cayman-gt4', 'cayman-gts',
                       'boxster-spyder', 'boxster-gts 4.0', 'boxster-gts',
                       'macan-gts', 'macan-turbo',
                       'cayenne-turbo s', 'cayenne-turbo gt', 'cayenne-turbo', 'cayenne-e-hybrid', 'cayenne-gts',
                       'panamera-gts', 'panamera-turbo', 'panamera-turbo s'
                       'taycan-turbo s', 'taycan-gts',
                       ]

    filtered_data = data[~data['trim'].isin(expensive_trims)]
    
    return filtered_data

def find_outliers_by_model(data, model, iqr_multiplier=1.5,only_lower_outliers=True):
    """
    Find outliers by year for a specific car model.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the data.
    - model (str): The specific car model to filter the data.
    - iqr (float): How many iqrs above or below the quartiles to consider outlier.

    Returns:
    - pandas.DataFrame: The table of outliers.
    """
    # Filter the data for the specific car model
    filtered_data = data[data['model'] == model]
    
    # Filter data by excluding expensive trims
    filtered_data = filter_expensive_trims(filtered_data)

    # Group the filtered data by year and calculate the price percentiles
    percentiles = filtered_data.groupby('year')['price'].quantile([0.25, 0.75]).unstack()

    # Create a new DataFrame to hold the outliers
    outliers = pd.DataFrame(columns=filtered_data.columns)
    
    # Iterate over each year and identify outliers
    for year in filtered_data['year'].unique():
        year_data = filtered_data[filtered_data['year'] == year]
        q1 = percentiles.loc[year, 0.25]
        q3 = percentiles.loc[year, 0.75]
        iqr = q3 - q1
        upper_threshold = q3 + iqr_multiplier * iqr + 500 # 1.5 x Q3
        lower_threshold = q1 - iqr_multiplier * iqr - 500 # 1.5 x Q1
        
        # If set, only return outliers that were priced significantly lower
        if only_lower_outliers:
            year_outliers = year_data[(year_data['price'] > upper_threshold) | (year_data['price'] < lower_threshold)]
        else: 
            year_outliers = year_data[(year_data['price'] < lower_threshold)]
        outliers = pd.concat([outliers, year_outliers])
    
    # Sort the outliers table by year and then by price in ascending order
    outliers = outliers.sort_values(['year', 'price'], ascending=[True, True])
    
    return outliers



def find_outliers(df):
    """
    Find the outliers for all car makes in the given DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing car data.

    Returns:
        pandas.DataFrame: A DataFrame containing the outliers
    """
    
    def process_car_make(car_make_df):
        # Process a specific car make DataFrame to find outliers for each model
        car_make_outliers = []
        for model in car_make_df['model'].unique():
            # Find the outliers for each model in the car make
            
            # If the number of rows for this model is fewer than 50, skip this model
            # Because the outlier detection wont work well
            if len(car_make_df)<50:
                continue
            
            model_outliers = find_outliers_by_model(car_make_df, model)
            car_make_outliers.append(model_outliers)

        return pd.concat(car_make_outliers)

    # Create an empty list to store the outlier DataFrames
    outliers = []

    for car_make in df['make'].unique():
        # Get the total data points for the current car make
        total_data_points = df[df['make'] == car_make]['model'].count()
        
        # Filter the DataFrame to get data for the current car make only
        car_make_df = df[df['make'] == car_make]
        
        try:
            # Process the current car make DataFrame to find outliers for each model
            car_make_outliers_df = process_car_make(car_make_df)
            
            # Append the outliers for the current car make to the outliers list
            outliers.append(car_make_outliers_df)
        except:
            # If there are no outliers for this make
            pass

    # Combine all the outlier DataFrames into one DataFrame
    outliers_df = pd.concat(outliers)
    
    return outliers_df



def remove_outliers_from_df(df):
    """
    Remove outliers from the input DataFrame. It first identifies the outliers using the `find_outliers` function.
    Then, it extracts the indexes of the outlier rows. Finally, it drops the rows corresponding to the outlier indexes and returns a new DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: A new DataFrame with outliers removed.

    """
    # Identify outliers
    outliers_df = find_outliers(df)

    # Extract the indexes of outlier rows
    outlier_indexes = outliers_df.index

    # Drop the rows corresponding to outlier indexes and create a new DataFrame
    cleaned_df = df.drop(outlier_indexes)
    
    # Report on how many outliers were removed
    rows_dropped = len(df) - len(cleaned_df)
    print(f"Removed {rows_dropped} outliers.")

    return cleaned_df



def finalize_dataframe(df, remove_outliers=True):
    """
    Make the final processing to the provided DataFrame.

    This function sets the correct data types for the 'odometer', 'price', 'transmission_manual', and 'year' columns.
    It sets the 'adIdUnique' column as the index and drops any rows with missing values.
    Removes any duplicate entries.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be processed.

    Returns:
    - pandas.DataFrame: The processed DataFrame with correct data types, index set, and missing values dropped.

    """
    # Set the correct data types for specified columns
    df = df.astype({"odometer": int, "price": int, "transmission_manual": int, "year": int})

    # Set 'adIdUnique' as the index
    df.set_index('adIdUnique', inplace=True)

    # Drop any rows with missing values
    df.dropna(inplace=True)
    
    # Check for duplicates
    nr_duplicate_entries = df.duplicated().sum()
    
    # Remove duplicate entries, keeping the last occurrence
    df.drop_duplicates(keep='last', inplace=True)
    
    print(f"Removed {nr_duplicate_entries} duplicates.")
    logging.info(f"INFO: Removed {nr_duplicate_entries} duplicates.")
    
    # Remove outliers from the DataFrame
    if remove_outliers:
        df = remove_outliers_from_df(df)
    
    # Drop fueltype column
    df = df.drop("fueltype",axis=1)
    
    print("Final row count:", len(df))
    logging.info(f"INFO: Final row count: {len(df)}")

    return df



def save_dataframe_to_s3(df, bucket, filename, save_index=False):
    """
    Save a DataFrame to a CSV file in an Amazon S3 bucket.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be saved.
    - bucket (str): The name of the Amazon S3 bucket.
    - filename (str): The desired filename for the CSV file.

    Returns:
    - None
    """

    # Convert DataFrame to CSV data
    csv_data = df.to_csv(index=save_index)

    # Create an S3 client
    s3 = boto3.client('s3')

    # Upload the CSV data to S3 bucket
    s3.put_object(Body=csv_data, Bucket=bucket, Key=filename)

    print("DataFrame saved to S3 bucket.")
    logging.info("SUCCESS: DataFrame saved to S3")
    
    
def save_file_to_s3(file_name, bucket, object_name=None):
    """
    Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name (if not specified then file_name is used)

    Returns:
    - None
    """
    # Create an S3 client
    s3 = boto3.client('s3')
    
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name    

    # Upload the file to S3 bucket
    try:
        response = s3.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        # If any error occurs during the upload, it will be printed out
        # and the function will return False indicating the upload failed.
        print(e)
        return False
    
    # If the upload is successful, the function returns True.
    print("File saved to S3 bucket.")
    logging.info("SUCCESS: File saved to S3")
    return True
    
    
def save_dataframe_to_csv(df, filename, save_index=False):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be saved.
    - filename (str): The desired filename for the CSV file.

    Returns:
    - None
    """

    df.to_csv(filename, index=save_index)
    print("DataFrame saved to CSV file.")
    logging.info("SUCCESS: DataFrame saved to CSV file.")
    
    
    
def create_trims_database(df):
    """
    Creates a trims database based on the provided DataFrame.

    This function takes a DataFrame as input, applies various operations to
    create a filtered database with the specified columns and conditions.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        pandas.DataFrame: A filtered database DataFrame with specified columns
                          and 'unknown' trims removed.
    """
    # Select only the desired columns from the DataFrame
    trims_df = df[['make','model','year','trim','bodytype','drivetrain']]

    # Remove rows with 'unknown' in the 'trim' column (case-insensitive)
    trims_df = trims_df[~trims_df['trim'].str.contains('unknown', case=False)]
    
    # Reset the index of the DataFrame
    trims_df = trims_df.reset_index(drop=True)

    # Remove duplicate rows, keeping the last occurrence
    trims_df = trims_df.drop_duplicates(keep='last')

    # Sort the DataFrame based on make, model, year, and trim
    trims_df = trims_df.sort_values(by=['make', 'model', 'year', 'trim'])
    
    # Convert 'model', 'trim' and 'drivetrain' columns to uppercase
    trims_df['model'] = trims_df['model'].str.upper()
    trims_df['trim'] = trims_df['trim'].str.upper()
    trims_df['drivetrain'] = trims_df['drivetrain'].str.upper()

    # Convert 'make' and 'bodytype' columns to proper case (capitalize each word)
    trims_df['make'] = trims_df['make'].str.title()
    trims_df['bodytype'] = trims_df['bodytype'].str.title()
    
    # Reset the index of the DataFrame after the operations
    trims_df = trims_df.reset_index(drop=True)
    
    trims_df.index.name = 'id'

    return trims_df
    
    

def preprocess_dataframe(df, trim_min_occurrences=10,model_min_occurrences=5, trim_combine_with_modelname=True, remove_outliers=True):
    """
    Preprocess the provided DataFrame by applying various data processing steps.

    This function performs the following processing steps in the specified order:
    1. Drops unnecessary columns.
    2. Cleans and standardizes the 'make' column.
    3. Cleans and standardizes the 'model' column.
    4. Removes infrequent trim levels.
    5. Cleans and standardizes the 'transmission' column.
    6. Cleans and standardizes the 'drivetrain' column.
    7. Cleans and standardizes the 'bodytype' column.
    8. Cleans and standardizes the 'province' column.
    9. Processes the 'odometer' column.
    10. Processes the 'price' column.
    11. Processes the 'year' column.
    12. Finalizes the DataFrame by dropping unnecessary columns.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be processed.

    Returns:
    - pandas.DataFrame: The processed DataFrame.

    """
    
    # Initial row count
    initial_rows = len(df)
    print("Initial row count:", initial_rows)
    
    # Filter dataframe
    print("Filtering some rows")
    df = drop_unnecessary_rows(df)
    print(f"--Row count:{len(df)}")
    
    # Drop unnecessary columns
    print("Dropping unnecessary columns")
    df = drop_unnecessary_columns(df)
    print(f"--Row count:{len(df)}")
    
    # Process 'make' column
    print("Processing make")
    df = process_make(df)
    print(f"--Row count:{len(df)}")

    # Process 'model' column
    print("Processing model")
    df = process_model(df, min_occurrences=model_min_occurrences)
    print(f"--Row count:{len(df)}")

    # Process 'trim' column
    print("Processing trim")
    df = process_trim(df, min_occurrences=trim_min_occurrences, combine_with_modelname=trim_combine_with_modelname)
    print(f"--Row count:{len(df)}")

    # Process 'transmission' column
    print("Processing transmission")
    df = process_transmission(df)
    print(f"--Row count:{len(df)}")

    # Process 'drivetrain' column
    print("Processing drivetrain")
    df = process_drivetrain(df)
    print(f"--Row count:{len(df)}")

    # Process 'bodytype' column
    print("Processing bodytype")
    df = process_bodytype(df)
    print(f"--Row count:{len(df)}")

    # Process 'province' column
    df = process_province(df)

    # Process 'odometer' column
    print("Processing odometer")
    df = process_odometer(df)
    print(f"--Row count:{len(df)}")

    # Process 'price' column
    print("Processing price")
    df = process_price(df)
    print(f"--Row count:{len(df)}")

    # Process 'year' column
    print("Processing year")
    df = process_year(df)
    print(f"--Row count:{len(df)}")
    
    # Finalize DataFrame
    print("Finalizing dataframe")
    df = finalize_dataframe(df, remove_outliers=remove_outliers)

    return df


if __name__ == "__main__":
    # Log start time
    import time
    start_time = time.time()

    # Import functions
    from preprocessing import *

    # Get the execution role (only in SageMaker)
    role = get_execution_role()

    # Declare bucket name, remote file, and destination
    my_bucket = 'hazar-ml-bucket'
    orig_file = 'carvalu/data/data.csv'
    dest_file = 'Data/raw/data.csv'

    # Download the file from S3
    download_file_from_s3(my_bucket, orig_file, dest_file)

    # Load the data
    df = read_csv_file(dest_file, index_col=0)

    # Preprocess the DataFrame
    df = preprocess_dataframe(df, trim_min_occurrences=5,model_min_occurrences=10, trim_combine_with_modelname=True)

    # Save to S3
    save_dataframe_to_s3(df, my_bucket, 'carvalu/data/data_clean.csv')

    # Save to local as csv
    save_dataframe_to_csv(df, 'Data/processed/data_clean.csv')

    # Create trims database out of data
    trims_database = create_trims_database(df)

    # Save trims database to csv file
    trims_database.to_csv('Data/other/trims_database.csv')

    # Save trims database to S3
    save_dataframe_to_s3(trims_database, my_bucket, 'carvalu/trims/trims_database.csv', save_index=True)

    # Log time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")