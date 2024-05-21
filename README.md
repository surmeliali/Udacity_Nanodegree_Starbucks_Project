# Starbucks Capstone Challenge

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Overview](#files)
4. [Solution Steps](#steps)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>
There are no necessary libraries to run the code, the Anaconda environment of Python will be enough. The code should run with no issues using Python versions 3.x. You can run the Jupyter Notebook (`Starbucks_Capstone_Challenge.ipynb`) in a compatible environment.

## Motivation <a name="motivation"></a>
This project explores the Starbucks Capstone Challenge, using a simulated dataset to analyze customer behavior on the Starbucks rewards mobile app. Starbucks aims to optimize offer targeting and customer engagement. The goal is to determine how different demographic groups respond to various types of offers.

We are interested to answer the following two questions:
1. Which offer should be sent to a particular customer to let the customer buy more?
2. Which demographic groups respond best to which offer type?


## Files Overview <a name="files"></a>
This data set is a simplified version of the real Starbucks app contains one underlying product among dozens of products Starbucks is selling.

### 1. `portfolio.json`
- Containing offer ids and meta data about each offer (duration, type, etc.)
- **Variables:**
  - `id` (string) - offer ID
  - `offer_type` (string) - type of offer (BOGO, discount, informational)
  - `difficulty` (int) - minimum required spend
  - `reward` (int) - reward for completing
  - `duration` (int) - time offer is open
  - `channels` (list of strings)

### 2. `profile.json`
- Demographic data for each customer.
- **Variables:**
  - `age` (int) - age of the customer
  - `became_member_on` (int) - date when customer created an app account
  - `gender` (str) - gender of the customer
  - `id` (str) - customer ID
  - `income` (float) - customer's income

### 3. `transcript.json`
- Records for transactions, offers received, viewed, and completed.
- **Variables:**
  - `event` (str) - record description
  - `person` (str) - customer ID
  - `time` (int) - time in hours
  - `value` (dict of strings) - offer ID or transaction amount

Feel free to adapt and enhance the project to suit your specific needs.


## Solution Approach <a name="steps"></a>

### 1. Data Cleaning
- Addressing issues related to users not opting into offers.
- Handling demographic variations and user purchases without receiving or viewing offers.

### 2. Data Analysis
- Exploratory Data Analysis (EDA) to uncover patterns and correlations.
- Identifying significant features influencing offer completion.

### 3. Model Building
- Utilizing machine learning models for predicting user responses.
- Assessing models such as RandomForest and GradientBoosting.

### 4. Results and Recommendations
- Presenting findings and actionable insights.
- Recommending strategies for effective offer targeting.


## Results <a name="results"></a>


The main findings of the code can be found at the post available [here](https://surmeliali.medium.com/get-in-line-to-take-the-best-offer-from-starbucks-f9b97ec591c8).

Our analysis leveraged machine learning techniques to predict offer completion and uncover demographic trends in customer responsiveness. Key determinants influencing target selection, including transaction amount, membership duration, starting time, and customer income, were identified through rigorous examination. Notably, transaction amount emerged as the most predictive feature, underscoring its critical role in delineating target demographics.

By employing thorough data preprocessing, training models with RandomForestClassifier, and fine-tuning parameters with GridSearchCV, we achieved an impressive 88.43% accuracy in predicting customer engagement with offers. Furthermore, we identified the demographic segments that show the most favorable responses to specific types of offers.Female respond notably higher response rates compared to men, whether it's for buy-one-get-one (BOGO) deals or discounts. Men tend to respond slightly more positively to discounts rather than BOGO offers. 







## Licancing, Authors, Acknowledgements <a name="licensing"></a>
Special thanks to Starbucks and Udacity for providing the data utilized in this project!