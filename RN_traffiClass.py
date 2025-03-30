import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configuración
IMG_SIZE = (64, 64)  # Tamaño de las imágenes
DATASET_DIR = 'dataset'  

# Función para cargar y preprocesar imágenes
def load_images(dataset_dir):
    images = []
    labels = []
    class_names = os.listdir(dataset_dir)  # Nombres de las carpetas (clases)
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
            if img is None:
                print(f"Error: No se pudo cargar la imagen {img_path}. Verifica la ruta o el formato.")
                continue
            img = cv2.resize(img, IMG_SIZE)  # Redimensionar
            img = img.flatten()  # Aplanar la imagen a un vector
            images.append(img)
            labels.append(class_names.index(class_name))  
    
    return np.array(images), np.array(labels), class_names

# Cargar imágenes
print("Cargando imágenes...")
images, labels, class_names = load_images(DATASET_DIR)
if len(images) == 0:
    print("No se encontraron imágenes. Verifica la ruta del dataset.")
    exit()
print("Imágenes cargadas:", images.shape)
print("Etiquetas cargadas:", labels.shape)
print("Clases:", class_names)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Entrenar un modelo SVM 
print("Entrenando modelo...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Función para predecir una nueva imagen
def predecir_señal(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
    if img is None:
        print(f"Error: No se pudo cargar la imagen {img_path}. Verifica la ruta o el formato.")
        return
    img = cv2.resize(img, IMG_SIZE) 
    img = img.flatten()  
    img = np.expand_dims(img, axis=0) 
    
    prediccion = model.predict(img)
    clase_predicha = class_names[prediccion[0]]
    print(f'Señal predicha: {clase_predicha}')


predecir_señal('pruebas/prueba.png')  