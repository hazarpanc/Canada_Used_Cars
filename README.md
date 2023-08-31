# Used Car Price Prediction

## Project Overview
This repository hosts a comprehensive machine learning pipeline that predicts the selling price of used cars based on various attributes. The pipeline encompasses data scraping, cleaning, preprocessing, feature engineering, model training, and deployment. Finally, the model was deployed on an AWS EC2 instance and is able to provide real-time predictions through my project website.

## Project Steps

## 1) Data Collection
**Data Collection**: I collected the data by web scraping multiple used car listing websites periodically over two years. I wrote a custom script to automate the scraping process, leveraging two popular Python libraries. Selenium to automate the web browser interaction, and BeautifulSoup to parse the HTML content and extract the required data points. I configured the script to run at bi-weekly intervals to collect the most up-to-date data available. It collected data from multiple pages and websites to build a comprehensive dataset of used car listings.

**Data Storage:** I adopted a strategic approach to manage the data collected during the weekly scraping sessions efficiently. The weekly scraped data was stored in individual CSV files, each representing one week's worth of data. This approach was not only flexible and scalable but also facilitated data backup and version control. 

I developed a script to merge the individual CSV files into a single comprehensive file. The script also handled any discrepancies in the data, such as duplicate entries or missing values. All the scraped data and the final merged dataset were stored in an AWS S3 bucket, which allowed for secure and scalable storage of the data.

## 2) Data Preprocessing
I wrote a preprocessing script to ensure that the collected data is cleaned, standardized and ready for model training. 

In the development of this preprocessing script, I adopted several best practices to ensure robustness and maintainability. Exception handling is incorporated to manage unexpected errors gracefully, and logging is used to facilitate debugging and monitoring. The code is modular, with functions, comments, and docstrings used to enhance readability and reusability. I unit tested the functions to make sure they behave as expected.

The preprocessing script includes various operations to clean and transform the data like standardization, missing value imputation, data transformation, outlier removal, mapping, filtering, sanity checks, data type conversion, and removing duplicates. 

- **Standardization**: 
  This included converting all string columns values to lowercase, standardizing transmission, body type values by applying specific mappings. Some examples to this are:
  - Converting all variations of "truck" like "cab", "super crew", "cutaway" to "truck").
  - On the drivetrain column, mapping values meaning all-wheel-drive like "4x4", "4X4", "4WD" to "AWD". 
  - The 'model' column was cleaned and standardized by applying specific mappings to fix variations and manually correct model names.

- **Missing Value Imputation**:
  Missing values in the 'drivetrain' column were filled with 'AWD', and missing values in the 'province' column were filled with 'ontario'.

- **Data Transformation**:
  The 'transmission' column was transformed into a new binary column 'transmission_manual' indicating whether the transmission is manual or not. The 'province' was extracted from the 'url' column. The 'odometer' and 'price' values were cleaned by removing units, commas, and converting them from string to integers. The 'year' column was processed to calculate the time difference from a reference date and the age of the car at the day the data was scraped.
  
- **Data Type Conversion**: Correct data types were set for the 'odometer', 'price', 'transmission_manual', and 'year' columns.

-  **Removing Duplicates**: Any duplicate entries in the dataframe were identified and removed, keeping the last occurrence.

- **Outlier Removal**:
Outliers were identified for each car model using a function that calculated price percentiles and then compared each price to an upper and lower threshold (1.5 times the interquartile range above the third quartile and below the first quartile, respectively). 

- **Filtering and Sanity Checks**:
Rows were filtered based on certain conditions, such as removing rows with year values exceeding the next year and filtering out rows with unreasonable price and odometer values. These sanity checks helped maintain data integrity by ensuring values are within reasonable and expected ranges.

- **Feature Engineering**: New features were engineered, such as 'days_since_reference' and 'car_age', to provide additional context and improve model performance.
- For more details on each of these steps, see the preprocess_dataframe function in the preprocessing.py script.
