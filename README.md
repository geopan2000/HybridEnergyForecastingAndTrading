# Forecasting and Trading in Renewable Energy

**Author**: Georgios Panagiotou  
**Email**: Giorgosp2000@gmail.com  
**University Email**: gsp23@ic.ac.uk  
**Project Title**: Forecasting and Trading in Renewable Energy

## Welcome to my Final Project for the MSc in Control and Optimization

### Abstract:
This project aims to enhance probabilistic models for solar and wind energy generation, focusing on the day-ahead electricity market. Using data from the Hornsea1 wind farm, East England’s PV generation, numerical weather predictions, and electricity market data, the study improves forecasting accuracy and optimizes trading strategies to maximize revenue under price uncertainties.

Based on the IEEE Hybrid Energy Forecasting and Trading Competition 2024, the project employed competition evaluation metrics. In the forecasting track, nine models were compared. A hybrid approach, independently predicting solar and wind using XGBoost regression (XGBR), followed by combining the outputs with another XGBoost model, achieved a 52% reduction in pinball loss compared to the benchmark. Other models, including LightGBM and Linear Regression, were also tested.

In the trading track, three decision-making algorithms were evaluated. The scenario-based optimization method, which predicted market prices in quantiles and generated 729 equally likely scenarios, yielded the best results. This approach increased revenue by 0.33% over the benchmark by optimizing expected revenue across different scenarios. The study demonstrated the highest pinball loss to revenue ratio during the competition’s offline test period, highlighting the effectiveness of the proposed forecasting and trading methods.

## GitHub Instructions:
1. **comp_utils.py**: Library provided by the HEFTcomp24, with some additional functions created by me.
2. **FT H-LGBM-1.ipynb**: Notebook for forecasting the Hybrid Energy Tune,Train,Test using LightGBM Regression model. Using NWP DATA.
3. **FT H-LGBM-2.ipynb**: Notebook for forecasting the Hybrid Energy Tune,Train,Test using LightGBM Regression model. Using the predictions from Notebooks 11, 14.
4. **FT H-LGBM-3.ipynb**: Notebook for forecasting the Hybrid Energy Tune,Train,Test using LightGBM Regression model. Using the predictions from Notebooks 11,12,13,14,15,16.
  
6. **FT H-LQR-1.ipynb**: Notebook for forecasting the Hybrid Energy Tune,Train,Test using Linear Regression regressor model. Using NWP DATA.
7. **FT H-LQR-2.ipynb**: Notebook for forecasting the Hybrid Energy Tune,Train,Test using Linear Regression regressor model. Using the predictions from Notebooks 12,15.
8. **FT H-LQR-3.ipynb**: Notebook for forecasting the Hybrid Energy Tune,Train,Test using Linear Regression regressor model. Using the predictions from Notebooks 11,12,13,14,15,16.
  
10. **FT H-XGBR-1.ipynb**: Notebook for forecasting the Hybrid Energy Tune,Train,Test using XGB Regression model. Using NWP DATA.
11. **FT H-XGBR-2.ipynb**: Notebook for forecasting the Hybrid Energy Tune,Train,Test using XGB Regression model. Using the predictions from Notebooks 13, 16.
12. **FT H-XGBR-3.ipynb**: Notebook for forecasting the Hybrid Energy Tune,Train,Test using XGB Regression model. Using the predictions from Notebooks 11,12,13,14,15,16.
  
14. **FT S-LGBM.ipynb**: Notebook for forecasting the Solar Energy Tune,Train,Test using LightGBM Regression model.
15. **FT S-LQR.ipynb**: Notebook for forecasting the Solar Energy Tune,Train,Test using Linear Quantile Regression model.
16. **FT S-XGBR.ipynb**: Notebook for forecasting the Solar Energy Tune,Train,Test using XGB regressor model.
    
18. **FT W-LGBM.ipynb**: Notebook for forecasting the Wind Energy Tune,Train,Test using LightGBM regressor model.
19. **FT W-LQR.ipynb**: Notebook for forecasting the Wind Energy Tune,Train,Test using LQR regressor model.
20. **FT W-XGB.ipynb**: Notebook for forecasting the Wind Energy Tune,Train,Test using XGB regressor model.
    
22. **TT DAP.ipynb**: Notebook for forecasting the Day Ahead Price.
23. **TT DASS.ipynb**: Notebook for forecasting the difference between DAP and SSP.
24. **TT NationalDemand.ipynb**: Notebook for forecasting the National Demand.
25. **TT SSP.ipynb**: Notebook for forecasting the Single System Price.
26. **Trading Track.ipynb**: In this notebook, we performed the scenario optimization strategy and calculated the revenue of all the other strategies.

27. **SolarTrainTable.ipynb**: Notebook for data preprocessing (on the training period) of the solar weather data.  
28. **SolarTestTable.ipynb**: Notebook for data preprocessing (on the test period) of the solar weather data.  
29. **WindTrainTable.ipynb**: Notebook for data preprocessing (on the training period) of the wind weather data.  
30. **WindTestTable.ipynb**: Notebook for data preprocessing (on the test period) of the wind weather data.  
31. **Data_Remit.ipynb**: Data preprocessing of availability data provided by REMIT.  
32. **DataToH5_20240129_20240519.ipynb**: An example of how we transformed the competition data into 2D dataframes.

### Data Access:
In order for the above notebooks to run, the data and models are required. I can provide the necessary data upon request. Please reach out to my personal email: giorgosp2000@gmail.com.

### Dependencies:
All required Python packages are listed in the requirements.txt
