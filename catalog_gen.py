import os
import pandas as pd

# Caminho base para a pasta predictions
base_dir = 'predictions'

# Criar uma lista para armazenar todos os dataframes
dataframes = []

# Percorrer todas as subpastas (moon e mars)
for planet in ['moon', 'mars']:
    planet_dir = os.path.join(base_dir, planet)
    
    # Percorrer todas as pastas dentro da pasta moon e mars
    for root, dirs, files in os.walk(planet_dir):
        for file in files:
            if file.endswith('.csv'):
                # Criar o caminho completo para o arquivo csv
                file_path = os.path.join(root, file)
                
                # Carregar o CSV em um DataFrame
                df = pd.read_csv(file_path)
                
                # Remover 'heatmaps_test_' e '.png' da coluna 'file'
                if 'file' in df.columns:
                    df['file'] = df['file'].str.replace('heatmap_test_', '', regex=False)
                    df['file'] = df['file'].str.replace('.png', '', regex=False)
                
                # Adicionar o dataframe à lista
                dataframes.append(df)

# Combinar todos os dataframes em um só
combined_df = (pd.concat(dataframes, ignore_index=True)).rename(columns={'predicted_output': 'time_rel(sec)'}).reset_index()[['file', 'time_rel(sec)']]

# Salvar o DataFrame combinado em um único arquivo CSV
combined_df.to_csv('catalog.csv', index=True)

print('Arquivo catalog.csv criado com sucesso.')