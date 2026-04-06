# Import libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings

# Load dataset in csv, competitor prices of inverters
warnings.simplefilter(action="ignore", category=FutureWarning)

# Resolve path relative to the script location for portability
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "price_competitor.csv")
df = pd.read_csv(data_path)

"""### Exploratory Data Analysis of Competitor Prices"""

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
has_missing = df.isnull().values.any()
print(f"Dataset has missing values: {has_missing}")
if has_missing:
    print(df.isnull().sum())

"""### Data Visualization"""

# sns.pairplot(df, x_vars='YEAR', y_vars=['SIEMENS_G120_055','SIEMENS_G120_075','SIEMENS_G120_22','VACON_20_055','VACON_20_075','VACON_20_22','Mitsubishi_FR_E_700_075','ABB_ACS150_075','ABB_ACS150_22','Danfoss_VLT_075','Danfoss_VLT_22'], kind="reg")
# Melt the dataframe to long format for easier plotting with FacetGrid
df_melted = df.melt(id_vars=["YEAR"], var_name="Company", value_name="Price(EURO)")

# Create a FacetGrid
g = sns.FacetGrid(df_melted, col="Company", col_wrap=4, height=4, aspect=1.5)
g.map(sns.regplot, "YEAR", "Price(EURO)")

# Adjust layout
plt.tight_layout()
plt.show()

# Histograms to check the normality assumption of the dependent variable (Prices)
# Create histograms with customized spacing
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))
df.hist(bins=20, ax=axes)

# Adjust spacing between plots
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

plt.show()

# Linear regression plots to visualize the relationship between YEAR and price for each inverter company
# Reuse the already melted data with regplot instead of 11 separate lmplot calls
g2 = sns.FacetGrid(df_melted, col="Company", col_wrap=4, height=4, aspect=1.5)
g2.map(sns.regplot, "YEAR", "Price(EURO)")
plt.tight_layout()
plt.show()

# Correlation Heatmap to check for multicollinearity among independent/dependent variables of companies

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmin=0, vmax=1, square=True, cmap="YlGnBu", ax=ax)
plt.show()

# Model Preparation

# Prepare features and target to predict years 2021-2023
X = df[["YEAR"]]
future_years = pd.DataFrame({"YEAR": np.arange(2021, 2024)})

# Dictionary to save predictions
predictions = {"YEAR": np.arange(2021, 2024)}

# Train a linear regression model for each column and predict future values of price
for column in df.columns[1:]:
    y = df[[column]]

    # Temporal split: train on earlier years, test on later years (avoids data leakage in time series)
    split_year = 2016
    train_mask = X["YEAR"] <= split_year
    test_mask = ~train_mask
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Train the model on training data
    model = LinearRegression().fit(X_train, y_train)

    # Evaluation the model on the test data
    score = model.score(X_test, y_test)
    print(f"R^2 score for {column}: {score:.2f}")
    # Print the model coefficients
    print(f"Coefficients for {column}: {model.coef_}")
    print(f"Intercept for {column}: {model.intercept_}")

    # Predict future values
    predictions[column] = model.predict(future_years).flatten()

# Convert all predictions to DataFrame
predictions_df = pd.DataFrame(predictions)

# Combine with the original data
df_combined = pd.concat([df, predictions_df], ignore_index=True)

# Show the predictions
print(df_combined)

# Plotting the prediction results
for column in df.columns[1:]:
    plt.figure()
    plt.plot(df["YEAR"], df[column], label="Historical Data")
    plt.plot(
        predictions_df["YEAR"],
        predictions_df[column],
        label="Predictions",
        linestyle="--",
    )
    plt.xlabel("Year")
    plt.ylabel(column)
    plt.title(f"Prediction for {column}")
    plt.legend()
    plt.show()
