# Import necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import regularizers

# Load the dataset from a CSV file
dataset = pd.read_csv('cancer.csv')

# Separate input features (x) and target variable (y)
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create a sequential neural network model
model = tf.keras.models.Sequential()

# Input layer with batch normalization and Leaky ReLU activation
model.add(tf.keras.layers.Dense(128, input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.BatchNormalization())  
model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

# Hidden layer with L2 regularization, batch normalization, and Leaky ReLU activation
model.add(tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001))) 
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LeakyReLU(alpha=0.01))

# Output layer with sigmoid activation for binary classification
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model with Adam optimizer, binary cross-entropy loss, and accuracy metric
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Define early stopping mechanism
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True
)

# Define ReduceLROnPlateau callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=8,
    min_lr=0.00001
)

# Train the model on the training data for 500 epochs and include early stopping and reduce_lr callbacks
history = model.fit(
    x_train, 
    y_train, 
    epochs=500, 
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model's performance on the test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict the test set results
y_pred = model.predict(x_test)
y_pred_binary = (y_pred > 0.5).astype("int").reshape(-1)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Print and plot the confusion matrix
print(cm)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print the model summary
model.summary()

# Save the trained model's weights and architecture to a file
model.save('cancerPredictor.h5')

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()