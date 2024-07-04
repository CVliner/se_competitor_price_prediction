# Import libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings

# Load dataset
warnings.simplefilter(action='ignore', category=FutureWarning)
os.getcwd()
df = pd.read_csv("/content/price_competitor.csv")

"""### Exploratory Data Analysis"""

# View the first few rows of the dataset

df.head()

# Get the column names of the dataset

df.columns

df

# Get the shape of the dataset (rows, columns)

df.shape

# Check information about the dataset, data types, and missing values

df.info()

# Get statistical summary of the numerical columns

df.describe().T

# Check for missing values in the dataset

df.isnull().values.any()
df.isnull().sum()

"""### Data Visualization"""

# sns.pairplot(df, x_vars='YEAR', y_vars=['SIEMENS_G120_055','SIEMENS_G120_075','SIEMENS_G120_22','VACON_20_055','VACON_20_075','VACON_20_22','Mitsubishi_FR_E_700_075','ABB_ACS150_075','ABB_ACS150_22','Danfoss_VLT_075','Danfoss_VLT_22'], kind="reg")
# Melt the dataframe to long format for easier plotting with FacetGrid
df_melted = df.melt(id_vars=['YEAR'], var_name='Company', value_name='Price(EURO)')

# Create a FacetGrid
g = sns.FacetGrid(df_melted, col="Company", col_wrap=4, height=4, aspect=1.5)
g.map(sns.regplot, 'YEAR', 'Price(EURO)')

# Adjust layout
plt.tight_layout()
plt.show()

# Histograms to check the normality assumption of the dependent variable (Sales)
# Create histograms with customized spacing
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
df.hist(bins=20,ax=axes)

# Adjust spacing between plots
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

plt.show()

# Linear regression plots to visualize the relationship between each independent variable and the dependent variable

sns.lmplot(x='SIEMENS_G120_055', y='YEAR', data=df)
sns.lmplot(x='SIEMENS_G120_075', y='YEAR', data=df)
sns.lmplot(x='SIEMENS_G120_22',y= 'YEAR', data=df)
sns.lmplot(x='VACON_20_055', y='YEAR', data=df)
sns.lmplot(x='VACON_20_075', y='YEAR', data=df)
sns.lmplot(x='VACON_20_22',y= 'YEAR', data=df)
sns.lmplot(x='Mitsubishi_FR_E_700_075', y='YEAR', data=df)
sns.lmplot(x='ABB_ACS150_075', y='YEAR', data=df)
sns.lmplot(x='ABB_ACS150_22',y= 'YEAR', data=df)
sns.lmplot(x='Danfoss_VLT_075', y='YEAR', data=df)
sns.lmplot(x='Danfoss_VLT_22',y= 'YEAR', data=df)

# Correlation Heatmap to check for multicollinearity among independent/dependent variables

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmin=0, vmax=1, square=True, cmap="YlGnBu", ax=ax)
plt.show()

# Model Preparation

# Prepare features and target
X = df[['YEAR']]
future_years = np.arange(2021, 2023).reshape(-1, 1)

# Dictionary to store predictions
predictions = {'YEAR': np.arange(2021, 2023)}

# Train a linear regression model for each column and predict future values
for column in df.columns[1:]:
    y = df[[column]]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model on training data
    model = LinearRegression().fit(X_train, y_train)

    # Optionally, you can evaluate the model on the test data
    score = model.score(X_test, y_test)
    print(f'R^2 score for {column}: {score:.2f}')
    # Print the model coefficients
    print(f'Coefficients for {column}: {model.coef_}')
    print(f'Intercept for {column}: {model.intercept_}')

    # Predict future values
    predictions[column] = model.predict(future_years).flatten()

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions)

# Combine with the original data
df_combined = pd.concat([df, predictions_df], ignore_index=True)

# Display the predictions
print(df_combined)

# Plotting the results
for column in df.columns[1:]:
    plt.figure()
    plt.plot(df['YEAR'], df[column], label='Historical Data')
    plt.plot(predictions_df['YEAR'], predictions_df[column], label='Predictions', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel(column)
    plt.title(f'Prediction for {column}')
    plt.legend()
    plt.show()
