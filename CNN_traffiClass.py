import sys, os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QTextEdit, QFileDialog, QLabel, QLineEdit)
from PyQt5.QtCore import Qt
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout# type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint# type: ignore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Entrenamiento CNN con Keras")
        self.resize(1000, 700)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.btn_seleccionar = QPushButton("Seleccionar carpeta base (train/val)")
        self.btn_seleccionar.clicked.connect(self.seleccionar_carpeta)
        layout.addWidget(self.btn_seleccionar)

        self.input_epochs = QLineEdit("40")
        layout.addWidget(QLabel("Número de épocas:"))
        layout.addWidget(self.input_epochs)

        self.input_batch = QLineEdit("32")
        layout.addWidget(QLabel("Tamaño del batch:"))
        layout.addWidget(self.input_batch)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.btn_entrenar = QPushButton("Entrenar CNN")
        self.btn_entrenar.clicked.connect(self.entrenar_modelo)
        layout.addWidget(self.btn_entrenar)

    def seleccionar_carpeta(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta base con train/val")
        if folder_path:
            self.folder_path = folder_path
            self.log_text.append(f"Carpeta seleccionada: {folder_path}")

    def entrenar_modelo(self):
        if not hasattr(self, 'folder_path'):
            self.log_text.append("Error: no has seleccionado la carpeta del dataset.")
            return

        try:
            epochs = int(self.input_epochs.text())
            batch_size = int(self.input_batch.text())
        except ValueError:
            self.log_text.append("Error: verifica que epochs y batch sean válidos.")
            return

        self.log_text.append("Cargando imágenes con data augmentation...")

        datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=20,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     brightness_range=[0.8, 1.2])

        train_generator = datagen.flow_from_directory(
            os.path.join(self.folder_path, "train"),
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='categorical'
        )

        val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            os.path.join(self.folder_path, "val"),
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='categorical'
        )

        num_classes = len(train_generator.class_indices)
        self.log_text.append(f"Clases detectadas: {train_generator.class_indices}")

        self.log_text.append("Construyendo modelo CNN...")

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.log_text.append("Entrenando modelo...")

        history = model.fit(train_generator,
                            epochs=epochs,
                            validation_data=val_generator)
        
        val_generator.reset()
        y_true = []
        y_pred = []

        for i in range(len(val_generator)):
            x_batch, y_batch = val_generator[i]
            preds = model.predict(x_batch)
            y_true.extend(np.argmax(y_batch, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        # Calcular matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys())

        self.log_text.append("\n--- Matriz de Confusión ---")
        self.log_text.append(str(cm))
        self.log_text.append("\n--- Reporte de Clasificación ---")
        self.log_text.append(report)

# 🔧 GRAFICAR curva de pérdida Y matriz de confusión en una sola figura
        self.fig.clf()
        self.fig.set_size_inches(12, 8)  # Ajustar tamaño correctamente

        # Subplot 1: Curva de pérdida
        ax1 = self.fig.add_subplot(211)
        ax1.plot(history.history['loss'], label='Entrenamiento', color='blue')
        ax1.plot(history.history['val_loss'], label='Validación', color='orange')
        ax1.set_title("Evolución de la pérdida")
        ax1.set_xlabel("Épocas")
        ax1.set_ylabel("Pérdida")
        ax1.legend()

        # Subplot 2: Matriz de confusión
        ax2 = self.fig.add_subplot(212)
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax2,
                    xticklabels=val_generator.class_indices.keys(),
                    yticklabels=val_generator.class_indices.keys())
        ax2.set_title("Matriz de Confusión")
        ax2.set_xlabel("Clase Predicha")
        ax2.set_ylabel("Clase Real")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)

        self.fig.tight_layout()
        self.fig.savefig("resultados_entrenamiento.png", dpi=300)  # ✅ Guardar imagen en disco
        self.canvas.draw()

        # Obtener precisión final de validación
        acc = history.history['val_accuracy'][-1]
        self.log_text.append(f"Precisión final de validación: {acc:.4f}")

        # Guardar modelo si tiene buena precisión
        if acc > 0.90:
            model.save("modelo_senales.keras")
            self.log_text.append("✅ Modelo guardado como 'modelo_senales.keras' 🎉")
        else:
            self.log_text.append("❌ Modelo no guardado. Precisión insuficiente (< 0.90)")

        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
