# example of Neural-SOM with Iris Dataset, edited by MSaavedra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Cargar el conjunto de datos Iris desde un archivo CSV
iris = pd.read_csv("iris3.csv")

# Obtener los datos y las etiquetas
data = iris.iloc[:, :-1].values  # 'data' son todas las columnas excepto la última
target = iris.iloc[:, -1].values  # 'target' es la última columna

# Define class labels
class_labels = {
    0: 'Set',
    1: 'Ver____',
    2: '___/Vir'
}

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Define the SOM parameters
x_dim = 10  # X-dimension of the SOM grid
y_dim = 10  # Y-dimension of the SOM grid
input_dim = data.shape[1]  # Input dimension (number of features)
sigma = 1.0  # Initial neighborhood radius
learning_rate = 0.5  # Initial learning rate
epochs = 20000  # Number of training epochs

# Initialize the SOM
som = MiniSom(x_dim, y_dim, input_dim, sigma=sigma, learning_rate=learning_rate)

# Initialize the weights
som.random_weights_init(data_normalized)

# Train the SOM
som.train(data_normalized, epochs)

# Create a hexagonal map
plt.figure(figsize=(10, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Plot the SOM distance map
plt.colorbar()

# Add markers for the data points with textual labels
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

for i, x in enumerate(data_normalized):
    w = som.winner(x)
    class_label = class_labels[target[i]]
    #plt.plot(w[0] + 0.5, w[1] + 0.5, markers[target[i]], markerfacecolor='None', markeredgecolor=colors[target[i]], markersize=10, markeredgewidth=2)
    plt.text(w[0], w[1], class_label, ha='left', va='bottom', fontsize=12, color='black')

plt.xticks(np.arange(0, x_dim, 1))
plt.yticks(np.arange(0, y_dim, 1))
plt.grid()
plt.title('Self-Organizing Map for Iris Dataset')
plt.show()
