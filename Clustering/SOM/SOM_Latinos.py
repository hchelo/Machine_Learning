# Example of SOM with LATAM Poverty map, according to CEPAL/ECLAC data of 2022
# Edited by MSaavedra
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

# Importamos el dataset para iniciar el análisis
poor = pd.read_csv("Datos_Pobreza.csv")

# Eliminar la primera columna "Id" si es necesario
poor = poor.drop('Id', axis=1)

# Obtener los datos y las etiquetas
data = poor.iloc[:, :-1].values  # 'data' son todas las columnas excepto la última
target = poor.iloc[:, -1].values  # 'target' es la última columna

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Define the SOM parameters
x_dim = 6  # X-dimension of the SOM grid
y_dim = 6  # Y-dimension of the SOM grid
input_dim = data.shape[1]  # Input dimension (number of features)
sigma = 3.0  # Initial neighborhood radius
learning_rate = 2.5  # Initial learning rate
epochs = 10000  # Number of training epochs

# Initialize the SOM
som = MiniSom(x_dim, y_dim, input_dim, sigma=sigma, learning_rate=learning_rate)

# Initialize the weights
som.random_weights_init(data_normalized)

# Train the SOM
som.train(data_normalized, epochs)

# Create a hexagonal map
plt.figure(figsize=(10, 8))
plt.pcolor(som.distance_map().T, cmap='gray')  # Plot the SOM distance map
plt.colorbar()

# Add markers for the data points with textual labels
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

for i, x in enumerate(data_normalized):
    w = som.winner(x)
    class_label = target[i]  # Utiliza el valor original como etiqueta
    plt.text(w[0], w[1], class_label, ha='left', va='bottom', fontsize=12, color='blue')

plt.xticks(np.arange(0, x_dim, 1))
plt.yticks(np.arange(0, y_dim, 1))
plt.grid()
plt.title('Self-Organizing Map - Case LATAM Poverty Map')
plt.show()
