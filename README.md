# Predict-the-trend-price-of-Google-stock-in-2017
According to the principle of the Brownian movement it is said that it is simply impossible to accurately predict stock prices in the future, so in our case we tried to predict not stock prices but rather the stock trends Google using an LSTM to find out if the trend is increasing or decreasing for Google stock.


To improve the model:

Five ways to improve the model:

1. Get more data: We trained the model over 5 years, but we could try with 10 years of data.

2. Increase the number of "timesteps": We chose 60 in the videos, which corresponds to about 3 months in the past. This number could be increased, for example to 120 (6 months).

3. Add other indicators: Do you know of other companies whose action could be correlated with that of Google? These actions could be added to the training data.

4. Add more LSTM layers: We already have 4 layers, but maybe with even more layers, we would get better results.

5. Add more neurons in each LSTM layer: We put 50 neurons in order to capture a certain complexity in the data, but is that enough? We could try with more neurons.

Obviously, these 5 ways can be combined together.
