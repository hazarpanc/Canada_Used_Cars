# Used Car Price Prediction
This repository contains the code for a machine learning model that predicts the selling price of used cars based on various attributes of the car. The model was trained on data scraped from used car listing websites, cleaned and preprocessed, and then used to train a LightGBM model. The model is deployed on an AWS EC2 instance and is able to provide real-time predictions.

## Project Overview
The project involves several key steps:

**1) Data Collection**: Data was collected by scraping used car listing websites at weekly intervals for a year.

**2) Data Preprocessing:** The collected data was cleaned and preprocessed using a series of steps to ensure that it was suitable for training the model. The preprocessing steps include:
- Dropping unnecessary columns.
- Cleaning and standardizing the 'make', 'model', 'trim', 'transmission', 'drivetrain', 'bodytype', and 'province' columns.
- Processing the 'odometer', 'price', and 'year' columns to remove outliers and convert the values to appropriate data types.
- Finalizing the DataFrame by dropping unnecessary columns.
- For more details on each of these steps, see the preprocess_dataframe function in the preprocessing.py script.

**3) Model Training:** A LightGBM model was trained on the preprocessed data. The features used for training are: 'make', 'model', 'odometer', 'bodytype', 'trim', 'year', 'drivetrain', 'fetchdate', 'transmission_manual', 'province', 'days_since_reference', 'car_age'. The target variable is 'price'.

**4) Model Deployment:** The trained model was deployed to an AWS EC2 instance, where it can provide real-time predictions of a car's selling price based on the given attributes.
