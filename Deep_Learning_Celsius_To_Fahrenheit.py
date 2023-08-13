import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#Training data
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Create the model -  A dense network
# Create a single layer with 1 input and 1 output
layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Once layers are defined, they need to be assembled into a model.
model = tf.keras.Sequential([layer_0])

# Compile the model with mean_sqared_error and Adam optimizer
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

#Train the model
#The cycle of calculate, compare, adjust is controlled by the fit method. The first 
#argument is the inputs, the second argument is the desired outputs. 
#The epochs argument specifies how many times this cycle should be run, and 
#the verbose argument controls how much output the method produces.
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

#Use the model to predict values
print(model.predict([100.0]))

#Graphically display training statistics
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()







