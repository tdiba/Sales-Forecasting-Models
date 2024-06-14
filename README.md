# Sales-Forecasting-Models

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
  - [Data Preparation](#data-preparation)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Model Selection and Training](#model-selection-and-training)
  - [Forecasting](#forecasting)
 
- [Key Deliverables](#key-deliverables)
- [Expected Outcomes](#expected-outcomes)
- [Tools and Technologies](#tools-and-technologies)
- [Conclusion](#conclusion)


### Project Overview

This project sought to develop accurate sales forecasting models to support inventory management, financial planning, and strategic decision-making for Langa Cash n Carry Stores. The models predict sales for various product categories and store locations over different time horizons (weekly, monthly, quarterly, annual).


### Data Sources

- Historical sales data from five store locations (Langa, Nyanga, Gugulethu, Pinelands, Thornton).
- External factors such as economic indicators and promotional periods.


### Methodology

#### 1. Data Preparation:

   - Imputed missing values using the median
   - Created additional features (day of the week, month, year)


#### 2. Exploratory Data Analysis (EDA):
    
    - Visualized sales trends over time
    - Analyzed sales distribution by day of the week and month

   
#### 3. Model Selection and Training:

    - Chose ARIMA model for time series forecasting
    - Trained the model on historical sales data and validated its accuracy

   
#### 4. Model Evaluation:

    - Evaluated model performance using RMSE
    - Compared actual vs. predicted sales values
   
#### 5. Forecasting:

    - Forecasting:
    - Visualized historical vs. forecasted sales data


  ### Key Deliverables

- Cleaned and enriched dataset.
- EDA visualizations.
- Trained ARIMA model.
- Evaluation metrics and visual comparison of actual vs. predicted sales.
- Sales forecasts for different time horizons.


### Expected Outcomes

- Improved inventory management.
- Enhanced financial planning and budgeting.
- Better promotional planning.
- Data-driven decision-making for strategic initiatives.


### Tools and Technologies

- Python (Pandas, NumPy, Statsmodels, Matplotlib)


### Conclusion

This project demonstrates the value of data-driven sales forecasting, providing actionable insights to optimize operations and improve business performance for Langa Cash n Carry Stores.

