# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

# Convert categorical to numeric
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Split features and labels
X = df.drop('charges', axis=1)
y = df['charges']

# Train-test split (80-20)
train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize data
scaler = StandardScaler()
train_dataset = scaler.fit_transform(train_dataset)
test_dataset = scaler.transform(test_dataset)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=[train_dataset.shape[1]]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mae',
    metrics=['mae']
)

# Train model
history = model.fit(
    train_dataset, train_labels,
    epochs=100,
    validation_split=0.2,
    verbose=0
)

# Evaluate model
loss, mae = model.evaluate(test_dataset, test_labels, verbose=2)
print("Mean Absolute Error:", mae)

# Prediction
predictions = model.predict(test_dataset).flatten()

# Plot
import matplotlib.pyplot as plt

plt.scatter(test_labels, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Health Costs')
plt.show()