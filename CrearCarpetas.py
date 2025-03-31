import os

# Lista de tus 20 clases
clases = [
    "Cruce Peatonal",
    "Zona Escolar",
    "Paso de Ciclistas",
    "Cruce de Ferrocarril",
    "Semaforo",
    "Doble Sentido",
    "Alto",
    "No cruzar",
    "Prohibido bicicletas",
    "No Estacionarse",
    "Prohibido arrojar basura",
    "Discapacidad",
    "Hospital",
    "Parada de autobus",
    "Vuelta Prohibida",
    "Limite de velocidad",
    "Prohibido seguir",
    "Prohibido celulares",
    "WC",
    "Uno x uno"
]

# Carpeta base donde se creará la estructura
base_path = "dataset"  # puedes cambiar esto si quieres

for tipo in ["train", "val"]:
    for clase in clases:
        path = os.path.join(base_path, tipo, clase)
        os.makedirs(path, exist_ok=True)

print("✅ Estructura de carpetas creada correctamente.")
