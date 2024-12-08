import pandas as pd

# Cargar los datos
data = pd.read_csv("../datasetoriginal/synthetic_energy_data.csv")

# Explorar los datos
print(data.head())  # Primeras filas
print(data.info())  # Información sobre tipos de datos
print(data.describe())  # Estadísticas descriptivas
