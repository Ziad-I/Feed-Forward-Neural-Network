import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from NN import NeuralNetwork, Layer

# Load the dataset
path = './concrete_data.xlsx'
df = pd.read_excel(path)

# Define features and targets
features = ['cement', 'water', 'superplasticizer', 'age']
targets = ['concrete_compressive_strength']
X = df[features]
y = df[targets]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# Standardize numerical features
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# need features and targets to be in shape (number of features/targets, m)
X_test = np.asarray(X_test.T)
X_train = np.asarray(X_train.T)
y_train = np.asarray(y_train.T)
y_test = np.asarray(y_test.T)

# todo: set the the parameters
layers = [
    Layer(inputs=4, neurons=32, activation="sigmoid", learning_rate=0.1),
    Layer(inputs=32, neurons=1, activation="linear", learning_rate=0.1)
]

nn = NeuralNetwork(layers)

nn.train(X_train, y_train)
A = nn.predict(X_test)

print("The error of the Neural Network model is: ", nn.error(y_test.flatten(), A.flatten()))
