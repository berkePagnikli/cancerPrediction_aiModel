import numpy as np
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('cancerPredictor.h5')
csvPath = "test.csv"
data = pd.read_csv(csvPath)
data = data.drop(columns=["diagnosis(1=m, 0=b)"]) 
predictions = model.predict(data)

print(predictions)