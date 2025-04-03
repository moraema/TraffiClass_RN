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
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import shutil

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

# Dentro de tu clase MainWindow

    def entrenar_modelo(self):
        self.entrenar_con_kfold(k_folds=5)

    def entrenar_con_kfold(self, k_folds=5):
        self.log_text.append(f"Iniciando validación cruzada con {k_folds} folds...")

        # Recolectar imágenes de train/ y val/
        all_images = []
        for folder in ['train', 'val']:
            base_path = os.path.join(self.folder_path, folder)
            for label in os.listdir(base_path):
                class_path = os.path.join(base_path, label)
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        all_images.append((os.path.join(class_path, file), label))

        df = pd.DataFrame(all_images, columns=['filepath', 'label'])
        df = df.sample(frac=1).reset_index(drop=True)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold = 1
        accuracies = []

        for train_idx, val_idx in skf.split(df['filepath'], df['label']):
            self.log_text.append(f"\n🔁 Fold {fold}/{k_folds}")
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            datagen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=20,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        brightness_range=[0.8, 1.2])

            train_gen = datagen.flow_from_dataframe(
                train_df, x_col='filepath', y_col='label',
                target_size=(128, 128),
                class_mode='categorical',
                batch_size=32
            )

            val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
                val_df, x_col='filepath', y_col='label',
                target_size=(128, 128),
                class_mode='categorical',
                batch_size=32
            )

            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(len(train_gen.class_indices), activation='softmax')
            ])

            model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

            history = model.fit(train_gen, epochs=15, validation_data=val_gen, verbose=0)

            acc = history.history['val_accuracy'][-1]
            accuracies.append(acc)
            self.log_text.append(f"📊 Precisión fold {fold}: {acc:.4f}")

            # Matriz de confusión y gráfica final SOLO del último fold
            if fold == k_folds:
                val_gen.reset()
                y_true, y_pred = [], []
                for i in range(len(val_gen)):
                    x_batch, y_batch = val_gen[i]
                    preds = model.predict(x_batch)
                    y_true.extend(np.argmax(y_batch, axis=1))
                    y_pred.extend(np.argmax(preds, axis=1))

                cm = confusion_matrix(y_true, y_pred)
                report = classification_report(y_true, y_pred, target_names=val_gen.class_indices.keys())

                self.log_text.append("\n--- Matriz de Confusión ---")
                self.log_text.append(str(cm))
                self.log_text.append("\n--- Reporte de Clasificación ---")
                self.log_text.append(report)

                self.fig.clf()
                self.fig.set_size_inches(12, 8)
                ax1 = self.fig.add_subplot(211)
                ax1.plot(history.history['loss'], label='Entrenamiento', color='blue')
                ax1.plot(history.history['val_loss'], label='Validación', color='orange')
                ax1.set_title("Evolución de la pérdida")
                ax1.set_xlabel("Épocas")
                ax1.set_ylabel("Pérdida")
                ax1.legend()

                ax2 = self.fig.add_subplot(212)
                sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax2,
                            xticklabels=val_gen.class_indices.keys(),
                            yticklabels=val_gen.class_indices.keys())
                ax2.set_title("Matriz de Confusión")
                ax2.set_xlabel("Clase Predicha")
                ax2.set_ylabel("Clase Real")
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)

                self.fig.tight_layout()
                self.fig.savefig("resultados_entrenamiento_kfold.png", dpi=300)
                self.canvas.draw()
                
                self.fig.clf()
                self.fig.set_size_inches(8, 4)
                ax = self.fig.add_subplot(111)
                ax.plot(range(1, k_folds + 1), accuracies, marker='o', color='green')
                ax.set_title("Precisión por Fold")
                ax.set_xlabel("Fold")
                ax.set_ylabel("Precisión")
                ax.set_ylim(0, 1)
                ax.grid(True)
                self.fig.tight_layout()
                self.fig.savefig("precision_kfold.png", dpi=300)
                self.canvas.draw()

                self.log_text.append("📈 Gráfica 'precision_kfold.png' guardada con éxito.")

            fold += 1

        promedio = np.mean(accuracies)
        self.log_text.append(f"\n✅ Precisión promedio validación cruzada: {promedio:.4f}")
        self.log_text.append("Entrenamiento finalizado.")
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
