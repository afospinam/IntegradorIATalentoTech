import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos
data = pd.read_csv("../datasetoriginal/synthetic_energy_data.csv")

# Consumo de energía a lo largo del tiempo
plt.figure(figsize=(15, 5))
plt.plot(data['Date'], data['Consumption_kWh'])
plt.title("Consumo de Energía a lo Largo del Tiempo")
plt.xlabel("Fecha")
plt.ylabel("Consumo (kWh)")
plt.show()

# Relación entre temperatura y consumo
plt.scatter(data['Temperature_C'], data['Consumption_kWh'], alpha=0.5)
plt.title("Relación entre Temperatura y Consumo de Energía")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Consumo (kWh)")
plt.show()
