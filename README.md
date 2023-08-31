# Used Car Price Prediction
This repository contains the code for a machine learning model that predicts the selling price of used cars based on various attributes of the car. The model was trained on data scraped from used car listing websites, cleaned and preprocessed, and then used to train a LightGBM model. The model is deployed on an AWS EC2 instance and is able to provide real-time predictions.

Currently the model is deployed on an AWS EC2 instance and is able to provide real-time predictions through my project website.

## Project Overview
The project involves several key steps:

**1) Data Collection**: I collected the data by web scraping multiple used car listing websites periodically over two years. I wrote a custom script to automate the scraping process, leveraging two popular Python libraries. Selenium to automate the web browser interaction, and BeautifulSoup to parse the HTML content and extract the required data points. I configured the script to run at bi-weekly intervals to collect the most up-to-date data available. It collected data from multiple pages and websites to build a comprehensive dataset of used car listings.

**Data Storage:** I adopted a strategic approach to manage the data collected during the weekly scraping sessions efficiently. The weekly scraped data was stored in individual CSV files, each representing one week's worth of data. This approach was not only flexible and scalable but also facilitated data backup and version control. 

I developed a script to merge the individual CSV files into a single comprehensive file. The script also handled any discrepancies in the data, such as duplicate entries or missing values. All the scraped data and the final merged dataset were stored in an AWS S3 bucket, which allowed for secure and scalable storage of the data.

**2) Data Preprocessing:** The collected data underwent a series of preprocessing steps to ensure its quality and suitability for model training. Key preprocessing steps included:
- Dropping unnecessary columns.
- **Cleaning and standardization:** The 'make', 'model', 'trim', 'transmission', 'drivetrain', 'bodytype', and 'province' columns were cleaned and standardized. For example, different variants of '4x4' in the 'drivetrain' column were all standardized to 'AWD'.
- Processing the 'odometer', 'price', and 'year' columns to remove outliers and convert the values to appropriate data types.
- **Feature Engineering**: New features were engineered, such as 'days_since_reference' and 'car_age', to provide additional context and improve model performance.
- For more details on each of these steps, see the preprocess_dataframe function in the preprocessing.py script.

