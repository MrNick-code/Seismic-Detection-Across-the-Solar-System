'''import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

local_dir = 'E:/Nasa Space Apps 2024/rn/rn'

def load_data(image_directory, csv_file):
    """
    Carrega as imagens (X) e monta o vetor de labels (y).
    
    Args:
        image_directory (str): Diretório das imagens.
        csv_file (str): Arquivo CSV contendo o nome da imagem e o valor de saída (y).

    Returns:
        X (np.array): Array de imagens pré-processadas.
        y (np.array): Array de labels correspondentes.
    """
    # Carregar o CSV com os valores de saída
    data = pd.read_csv(csv_file)

    X = []
    y = []

    # Iterar sobre cada linha do CSV
    for index, row in data.iterrows():
        image_name = row['file']
        target_value = row['y']

        # Caminho completo da imagem
        image_path = os.path.join(image_directory, 'heatmap_'+image_name+'.png')

        # Carregar a imagem
        image = cv2.imread(image_path)  # Garantindo que a imagem é lida em escala de cinza

        # Redimensionar a imagem (opcional, já que você mencionou que elas estão em 128x128)
        image_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)

        # Normalizar os valores da imagem para [0, 1]
        image_normalized = image_resized / 255.0

        # Adicionar a imagem ao array X
        X.append(image_normalized)

        # Adicionar o valor de saída ao array y
        y.append(target_value)

    # Converter X e y para arrays NumPy
    X = np.array(X)
    y = np.array(y)

    # Adicionar a dimensão do canal (X precisa estar no formato (n_samples, altura, largura, 1))
    X = np.expand_dims(X, axis=-1)  # Adicionando a dimensão do canal

    return X, y

# Diretório de imagens e arquivo CSV com os labels
image_directory = f"{local_dir}/heatmaps_pre_processed"
csv_file = f"{local_dir}/arrival_times.csv"

# Carregar os dados
X, y = load_data(image_directory, csv_file)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)

def build_regression_model(input_shape):
    model = models.Sequential()

    # Primeira camada convolucional + pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Segunda camada convolucional + pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Flatten (transformar a saída 2D em 1D para alimentar a camada densa)
    model.add(layers.Flatten())

    # Camadas densas totalmente conectadas
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))

    # Camada de saída para regressão (uma única saída)
    model.add(layers.Dense(1))

    return model

# Definir o formato da imagem de entrada
input_shape = (64, 64, 3)  # (altura, largura, canais)

# Construir o modelo
model = build_regression_model(input_shape)

# Compilar o modelo com um otimizador adequado para regressão
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Resumo da arquitetura da rede
model.summary()

# Definir parâmetros de treinamento
epochs = 100
batch_size = 32

#early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Treinar o modelo
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data = (X_val, y_val),
    callbacks=[checkpoint]
)

model.save('model.h5')

# Fazer previsões
y_pred = model.predict(X_test).flatten()

# Plotar valores reais vs. previstos
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.title("Valores Reais vs. Previsto")
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r')  # Linha de referência
plt.grid(True)
plt.savefig('RxP.png')
plt.show()

plt.figure(figsize=(12, 5))

plt.plot(history.history['loss'][1:], label='Training Loss')
plt.plot(history.history['val_loss'][1:], label='Valitation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Training and Validation Loss.png')
plt.show()'''

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

# Definindo a função personalizada R^2
def r2_score_tf(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))  # Soma dos quadrados dos resíduos
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Soma total dos quadrados
    return 1 - ss_res / (ss_tot + K.epsilon())  # Epsilon para evitar divisão por zero

local_dir = 'E:/Nasa Space Apps 2024/rn/rn'

def load_data(image_directory, csv_file):
    """
    Carrega as imagens (X) e monta o vetor de labels (y).
    
    Args:
        image_directory (str): Diretório das imagens.
        csv_file (str): Arquivo CSV contendo o nome da imagem e o valor de saída (y).

    Returns:
        X (np.array): Array de imagens pré-processadas.
        y (np.array): Array de labels correspondentes.
    """
    # Carregar o CSV com os valores de saída
    data = pd.read_csv(csv_file)

    X = []
    y = []

    # Iterar sobre cada linha do CSV
    for index, row in data.iterrows():
        image_name = row['file']
        target_value = row['y']

        # Caminho completo da imagem
        image_path = os.path.join(image_directory, 'heatmap_'+image_name+'.png')

        # Carregar a imagem
        image = cv2.imread(image_path)  # Garantindo que a imagem é lida em escala de cinza

        # Redimensionar a imagem (opcional, já que você mencionou que elas estão em 128x128)
        image_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)

        # Normalizar os valores da imagem para [0, 1]
        image_normalized = image_resized / 255.0

        # Adicionar a imagem ao array X
        X.append(image_normalized)

        # Adicionar o valor de saída ao array y
        y.append(target_value)

    # Converter X e y para arrays NumPy
    X = np.array(X)
    y = np.array(y)

    # Adicionar a dimensão do canal (X precisa estar no formato (n_samples, altura, largura, 1))
    X = np.expand_dims(X, axis=-1)  # Adicionando a dimensão do canal

    return X, y

# Diretório de imagens e arquivo CSV com os labels
image_directory = f"{local_dir}/heatmaps_pre_processed"
csv_file = f"{local_dir}/arrival_times.csv"

# Carregar os dados
X, y = load_data(image_directory, csv_file)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)

def build_regression_model(input_shape):
    model = models.Sequential()

    # Primeira camada convolucional + pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Segunda camada convolucional + pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Flatten (transformar a saída 2D em 1D para alimentar a camada densa)
    model.add(layers.Flatten())

    # Camadas densas totalmente conectadas
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))

    # Camada de saída para regressão (uma única saída)
    model.add(layers.Dense(1))

    return model

# Definir o formato da imagem de entrada
input_shape = (64, 64, 3)  # (altura, largura, canais)

# Construir o modelo
model = build_regression_model(input_shape)

# Compilar o modelo com um otimizador adequado para regressão
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r2_score_tf])

# Resumo da arquitetura da rede
model.summary()

# Definir parâmetros de treinamento
epochs = 150
batch_size = 32

# Treinar o modelo
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data = (X_val, y_val)
)

model.save('model.h5')

# Fazer previsões
y_pred = model.predict(X_test).flatten()

# Plotar valores reais vs. previstos
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.title("Valores Reais vs. Previsto")
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r')  # Linha de referência
plt.grid(True)
plt.savefig('RxP.png')
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)

plt.plot(history.history['loss'][1:], label='Training Loss')
plt.plot(history.history['val_loss'][1:], label='Valitation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1,2,2)

plt.plot(history.history['r2_score_tf'][1:], label='Training R²')
plt.plot(history.history['val_r2_score_tf'][1:], label='Validation R²')
plt.xlabel('Epochs')
plt.ylabel('R² Score')
plt.title('Training and Validation R² Score')
plt.legend()
plt.savefig('Training and Validation Loss.png')

plt.show()

def load_data(image_directory):
    """
    Carrega as imagens (X) e monta o vetor de labels (y).
    
    Args:
        image_directory (str): Diretório das imagens.
        csv_file (str): Arquivo CSV contendo o nome da imagem e o valor de saída (y).

    Returns:
        X (np.array): Array de imagens pré-processadas.
        y (np.array): Array de labels correspondentes.
    """
    # Carregar o CSV com os valores de saída

    data = os.listdir(image_directory)

    X = []

    # Iterar sobre cada linha do CSV
    for image_name in data:

        # Caminho completo da imagem
        image_path = os.path.join(image_directory, image_name)

        # Carregar a imagem
        image = cv2.imread(image_path)  # Garantindo que a imagem é lida em escala de cinza

        # Redimensionar a imagem (opcional, já que você mencionou que elas estão em 128x128)
        image_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)

        # Normalizar os valores da imagem para [0, 1]
        image_normalized = image_resized / 255.0

        # Adicionar a imagem ao array X
        X.append(image_normalized)

    # Converter X e y para arrays NumPy
    X = np.array(X)

    # Adicionar a dimensão do canal (X precisa estar no formato (n_samples, altura, largura, 1))
    X = np.expand_dims(X, axis=-1)  # Adicionando a dimensão do canal

    return X, data

local_dir = 'E:/Nasa Space Apps 2024/rn/rn'

for [body, body_type] in [['moon', 'lunar'], ['mars', 'mars']]:
    if body == 'moon':
        for probe in ['/S12_GradeB', '/S15_GradeA','/S15_GradeB', '/S16_GradeA', '/S16_GradeB']:
            image_directory = f"{local_dir}/heatmaps_test_{body}_pre_processed{probe}"
            durations = np.array((pd.read_csv(f"{local_dir}/durations/{body}{probe}{probe}durations.csv"))['duration'])

            # Carregar os dados
            X_test, data = load_data(image_directory)

            # Fazer previsões
            y_pred = model.predict(X_test).flatten()

            # Criar DataFrame com os nomes dos arquivos e as previsões
            results_df = pd.DataFrame({
                'file': data,
                'predicted_output': y_pred*durations
            })

            # Salvar o DataFrame em um arquivo CSV
            results_df.to_csv(f"{local_dir}/predictions/{body}{probe}{probe}_predictions.csv", index=True)
            print("CSV file 'predictions.csv' saved successfully.")
    
    elif body == 'mars':
        probe = ''
        image_directory = f"{local_dir}/heatmaps_test_{body}_pre_processed{probe}"
        durations = np.array((pd.read_csv(f"{local_dir}/durations/{body}/{body}_durations.csv"))['duration'])

        # Carregar os dados
        X_test, data = load_data(image_directory)

        # Fazer previsões
        y_pred = model.predict(X_test).flatten()

        # Criar DataFrame com os nomes dos arquivos e as previsões
        results_df = pd.DataFrame({
            'file': data,
            'predicted_output': y_pred*durations
        })

        # Salvar o DataFrame em um arquivo CSV
        results_df.to_csv(f"{local_dir}/predictions/{body}/{body}_predictions.csv", index=True)
        print("CSV file 'predictions.csv' saved successfully.")