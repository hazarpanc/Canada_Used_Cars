# Used Car Price Prediction

## Project Overview
Predicting the selling price of used cars is a critical task for both buyers and sellers in the pre-owned car market. Overpriced cars can lead to lost sales, while underpriced cars can lead to financial losses for the seller. This project aims to develop a machine learning model to predict used car prices accurately, helping both buyers and sellers make informed decisions.

This repository contains the machine learning pipeline I built for this purpose. The pipeline encompasses data scraping, cleaning, preprocessing, feature engineering, model training, and deployment. Finally, the model was deployed on an AWS EC2 instance and is able to provide real-time predictions through my project website.

![Website](https://carvalu.ca/images/website_ss3.jpg)

## Project Steps

![Process](https://carvalu.ca/images/process_pipeline.png)

## 1) Data Collection
**Data Collection**: I collected the data by web scraping multiple used car listing websites periodically over two years. I wrote a custom script to automate the scraping process, leveraging two Python libraries: Selenium, for automating web browser interaction, and BeautifulSoup, for parsing HTML content and extracting required data points. I configured the script to run at bi-weekly intervals to collect the most up-to-date data available. It collected data from multiple pages and websites to build a comprehensive dataset of used car listings.

**Data Storage:** I adopted a strategic approach to manage the data collected during the weekly scraping sessions efficiently. The weekly scraped data was stored in individual CSV files, each representing one week's worth of data. This approach was not only flexible and scalable but also facilitated data backup and version control. 

I developed a script to merge the individual CSV files into a single comprehensive file. The script also handled any discrepancies in the data, such as duplicate entries or missing values. All the scraped data and the final merged dataset were stored in an AWS S3 bucket, which allowed for secure and scalable storage of the data.

## 2) Data Preprocessing
I wrote a preprocessing script to ensure that the collected data is cleaned, standardized and ready for model training. 

In the development of this preprocessing script, I adopted several best practices to ensure robustness and maintainability. Exception handling is incorporated to manage unexpected errors gracefully, and logging is used to facilitate debugging and monitoring. The code is modular, with functions, comments, and docstrings used to enhance readability and reusability. I unit tested the functions to make sure they behave as expected.

The preprocessing script includes various operations to clean and transform the data, such as standardization, missing value imputation, data transformation, outlier removal, mapping, filtering, sanity checks, data type conversion, and removing duplicates.

- **Standardization**: 
  This included converting all string columns values to lowercase, standardizing transmission, body type values by applying specific mappings. Some examples to this are:
  - Converting all variations of "truck" like "cab", "super crew", "cutaway" to "truck").
  - On the drivetrain column, mapping values meaning all-wheel-drive like "4x4", "4X4", "4WD" to "AWD". 
  - The 'model' column was cleaned and standardized by applying specific mappings to fix variations and manually correct model names.

- **Missing Value Imputation**:
  Missing values in the 'drivetrain' column were filled with 'AWD', and missing values in the 'province' column were filled with 'ontario'.

- **Data Transformation**:
  The 'odometer' and 'price' values were cleaned by removing units, commas, and converting them from string to integers. The 'year' column was processed to calculate the time difference from a reference date and the age of the car at the day the data was scraped.

- **Data Type Conversion**: Correct data types were set for the 'odometer', 'price', 'transmission_manual', and 'year' columns.

-  **Removing Duplicates**: Any duplicate entries in the dataframe were identified and removed, keeping the last occurrence.

- **Outlier Removal**:
Outliers were identified for each car model using a function that calculated price percentiles and then compared each price to an upper and lower threshold (1.5 times the interquartile range above the third quartile and below the first quartile, respectively).

![Outliers](https://carvalu.ca/images/outliers.jpg)

- **Filtering and Sanity Checks**:
Rows were filtered based on certain conditions, such as removing rows with year values exceeding the next year and filtering out rows with unreasonable price and odometer values. These sanity checks helped maintain data integrity by ensuring values are within reasonable and expected ranges.

- **Feature Engineering:**
  New features were engineered to improve model performance.
  - 'days_since_reference' was calculated by determining the time difference between a reference date and the day the data was scraped. I created this feature for the model to capture the changes in price trends in the used car market.
  - 'car_age' was created by calculating the number of years since the car's model year, including decimals, to provide additional context and improve model performance.
  - The 'transmission' column was transformed into a new binary column 'transmission_manual' indicating whether the transmission is manual or not.
  - The 'province' of the listing was extracted from the 'url' column. 

For more details on each of these steps, see the the preprocessing.py script.


Before Preprocessing:
![Before](https://carvalu.ca/images/before.jpg)

After Preprocessing:
![After](https://carvalu.ca/images/after.jpg)

## 3) Model Training 

Several models were experimented with during the development phase, including XGBoost. However, LightGBM was ultimately chosen due to its speed and efficiency advantages over XGBoost in terms of both memory usage and computation speed.

The final model was trained using LightGBM Regression. The features used for training the model are: 'make', 'model', 'odometer', 'bodytype', 'trim', 'year', 'drivetrain', 'fetchdate', 'transmission_manual', 'province', 'days_since_reference', 'car_age'. The target variable is 'price'.

The script executes the following steps:

1.  **Data Loading**: Loads a cleaned dataset from a CSV file. The dataset contains various features of used cars which will be used to predict the prices.
2.  **Data Encoding**: Encodes the categorical features using target encoding and ordinal encoding. This is a crucial step as it converts categorical data into a format that can be fed into the model.
3.  **Data Splitting**: Splits the data into training, validation, and test sets based on the fetch date. This ensures that the model is trained and tested on different sets of data to avoid overfitting.
4.  **Model Training**: Train a LightGBM model using the training and validation sets. 
5.  **Model Evaluation**: Evaluates the model on the test set. This step is crucial to understand how well the model will perform on unseen data.
6.  **Hyperparameter Optimization**: Optimizes the hyperparameters of the LightGBM model using Optuna (hyperparameter optimization framework).
7.  **Final Model Training**: Trains the final LightGBM model using the optimized hyperparameters and the whole dataset. This ensures that the model is trained on the most amount of data possible, thereby improving its prediction capability.
8.  **Model Saving**: Saves the trained model, target encoder, and ordinal encoder to files. This step ensures that the model and encoders can be reused in the future without retraining.

**Best Practices**

Best practices have been followed throughout the development of this script. 
- Exception handling has been used wherever necessary to ensure that the script does not crash due to minor issues.
- Logging has been used to record the flow of the script and any issues that may arise. This ensures that any issues can be easily tracked and resolved.
- The code is well-documented with comments and docstrings to ensure it is easy to understand and modify.
- Other best practices include using functions for code modularity and reusability, and following the PEP 8 style guide for Python code.

## 4) Model Deployment: 
I deployed the trained model to an AWS EC2 instance as an API, and it is able to provide real-time predictions of a car's selling price based on the given attributes. The API is served as a web application through my project website carvalu.ca

## Results

The model achieved a Mean Absolute Percentage Error (MAPE) of 5% on the test data, which is a satisfactory performance considering the nature of the used car listings data. The data is inherently noisy and contains several unobserved factors that can significantly affect the car prices. For example, the condition of the car, any previous accidents, and the fairness of the dealer's pricing strategy, all play a crucial role in determining the car's price. However, not all of this information is available in the dataset. Given these challenges and unobserved factors, a 5% MAPE indicates that the model is capable of making reasonably accurate predictions on the car prices.

## Technologies and Frameworks Used in This Project
![Skills](https://carvalu.ca/images/skills_map.png)
