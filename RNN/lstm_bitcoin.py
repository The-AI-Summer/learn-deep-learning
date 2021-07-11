from google.colab import files
files.upload()
df = pd.read_csv("market-price.csv",header=None)
dates=df['Date']
df.drop(['Date'], 1, inplace=True)
df.head()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units= 100,activation= 'tanh'.input_shape=(None, 1)))
model.add(tf.keras.layers.Dropout(rate= 0.2))
model.add(tf.keras.layers.Dense(units= 1,  activation= 'linear'))
model.compile(optimizer= 'adam', loss= 'mse')


model.fit(x=x_train,y=y_train,batch_size= 1,epochs= 100,verbose=True );
# Epoch 100/100
# 164/164 [==============================] - 1s 4ms/step - loss: 0.0020
inputs = min_max_scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_price = model.predict(inputs)
plt.plot(dates[len(df)-prediction_days:],test_set[:, 0], color='red', label='Real BTC Price')
plt.plot(dates[len(df)-prediction_days:],predicted_price[:, 0], color = 'blue', label = 'Predicted BTC Price')