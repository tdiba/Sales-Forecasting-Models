#!/usr/bin/env python
# coding: utf-8

# ### 1. Data Processing

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#Load Data
df=pd.read_excel(r"C:\Users\USER\Documents\Data Portfolio Projects\Retail\Sales Forecasting\refined_sales_forecasting_dataset.xlsx")
df.head()


# In[3]:


# Handle missing values by imputing with the median
df.fillna(df.median(), inplace=True)


# In[4]:


#Create additional features that may help in forecasting, such as day of the week, month, and year

df['Day_of_Week'] = df['Date'].apply(lambda x: pd.Timestamp(x).dayofweek)
df['Month'] = df['Date'].apply(lambda x: pd.Timestamp(x).month)
df['Year'] = df['Date'].apply(lambda x: pd.Timestamp(x).year)


# In[ ]:





# ### Exploratory Data Analysis

# Data will be visualized to identify trends, seasonality, and relationships between variables.

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# Plot of the sales trends for the Langa Store and the Fresh Produce category
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Langa_Fresh Produce'], label='Langa Fresh Produce')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Trend for Langa Fresh Produce')
plt.legend()
plt.show()


# In[7]:


# Sales by day of the week
plt.figure(figsize=(14, 7))
sns.boxplot(x='Day_of_Week', y='Langa_Fresh Produce', data=df)
plt.xlabel('Day of the Week')
plt.ylabel('Sales')
plt.title('Sales Distribution by Day of the Week for Langa Fresh Produce')
plt.show()


# In[8]:


# Sales by month
plt.figure(figsize=(14, 7))
sns.boxplot(x='Month', y='Langa_Fresh Produce', data=df)
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Distribution by Month for Langa Fresh Produce')
plt.show()


# In[ ]:





# ### Model Selection and Training

# We'll use a simple ARIMA model for time series forecasting. We'll start by preparing the data for the ARIMA model and then train the model.

# ##### Preparing the Data

# In[9]:


from statsmodels.tsa.arima.model import ARIMA


# In[10]:


# Convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'])


# In[11]:


# Set the date column as the index
df.set_index('Date', inplace=True)


# In[12]:


# Select the time series for the Langa store and Fresh Produce category
series = df['Langa_Fresh Produce']


# In[13]:


# Split the data into training and testing sets
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]


# In[ ]:





# #### Train the ARIMA Model

# In[14]:


# Train the ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Print the summary of the model
print(model_fit.summary())


# In[ ]:





# ### Model Evaluation

# In[15]:


# Make predictions
predictions = model_fit.forecast(steps=len(test))


# In[16]:


# Plot the actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Sales for Langa Fresh Produce')
plt.legend()
plt.show()


# In[17]:


# Calculate evaluation metrics
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')


# In[ ]:





# ### Forecasting

# Finally, we'll generate forecasts for different time horizons (weekly, monthly, quarterly, annual) and visualize the results

# In[18]:


# Generate forecasts for the next year
future_steps = 365
forecast = model_fit.forecast(steps=future_steps)


# In[19]:


# Plot the forecast
plt.figure(figsize=(14, 7))
plt.plot(series.index, series, label='Historical')
plt.plot(pd.date_range(start=series.index[-1], periods=future_steps, freq='D'), forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast for Langa Fresh Produce')
plt.legend()
plt.show()


# In[ ]:




