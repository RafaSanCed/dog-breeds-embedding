import requests
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.spatial import distance
import plotly.graph_objects as go

# Función para generar bigramas de una palabra
def generate_bigrams(word):
    return [word[i:i+2] for i in range(len(word) - 1)]

# Función para obtener datos de razas de perros de The Dog API
def fetch_dog_breeds(limit=100):
    url = 'https://api.thedogapi.com/v1/breeds'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        titles = [breed['name'] for breed in data[:limit]]
        return titles
    else:
        print("Error fetching data from The Dog API")
        return []

# Obtener nombres de razas de perros
titles = fetch_dog_breeds(100)

# Crear un conjunto de todos los bigramas en la base de datos
all_bigrams = set()
for title in titles:
    bigrams = generate_bigrams(title)
    all_bigrams.update(bigrams)

# Crear un diccionario de bigramas con índices únicos
bigram_to_index = {bigram: idx for idx, bigram in enumerate(all_bigrams)}

# Crear una función para generar el vector de bigramas para una palabra
def generate_bigram_vector(word, bigram_to_index):
    bigrams = generate_bigrams(word)
    vector = [0] * len(bigram_to_index)
    for bigram in bigrams:
        if bigram in bigram_to_index:
            vector[bigram_to_index[bigram]] = 1
    return vector

# Generar los vectores de bigramas para todos los nombres
title_vectors = {title: generate_bigram_vector(title, bigram_to_index) for title in titles}

# Convertir el diccionario de embeddings a un DataFrame
title_vectors_df = pd.DataFrame.from_dict(title_vectors, orient='index')

# Aplicar PCA para reducir los embeddings a 3 dimensiones
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(title_vectors_df)

# Crear un DataFrame con las dimensiones reducidas
reduced_df = pd.DataFrame(reduced_embeddings, columns=['PC1', 'PC2', 'PC3'])
reduced_df['title'] = title_vectors_df.index

# Visualizar los embeddings reducidos utilizando Plotly
fig = px.scatter_3d(reduced_df, x='PC1', y='PC2', z='PC3', text='title', title='Embeddings Visualization')
fig.update_traces(marker=dict(size=3))

# Función para encontrar el punto más cercano
def find_closest_point(new_vector, reduced_embeddings, titles):
    distances = [distance.euclidean(new_vector, vec) for vec in reduced_embeddings]
    min_index = np.argmin(distances)
    return titles[min_index], distances[min_index], min_index

# Función para agregar un nuevo punto y encontrar el más cercano
def add_and_visualize_new_text(new_text):
    new_vector = generate_bigram_vector(new_text, bigram_to_index)
    new_reduced_vector = pca.transform([new_vector])[0]
    
    closest_title, closest_distance, closest_index = find_closest_point(new_reduced_vector, reduced_embeddings, reduced_df['title'])
    
    new_point = pd.DataFrame([new_reduced_vector], columns=['PC1', 'PC2', 'PC3'])
    new_point['title'] = new_text
    
    # Agregar el nuevo punto en rojo
    fig.add_scatter3d(x=new_point['PC1'], y=new_point['PC2'], z=new_point['PC3'], text=new_point['title'], mode='markers', marker=dict(size=5, color='red'))
    
    # Agregar el punto más cercano en verde
    closest_point = reduced_df.iloc[closest_index]
    fig.add_scatter3d(x=[closest_point['PC1']], y=[closest_point['PC2']], z=[closest_point['PC3']], text=[closest_point['title']], mode='markers', marker=dict(size=5, color='green'))
    
    # Agregar un círculo alrededor del nuevo punto
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    x_circle = new_reduced_vector[0] + closest_distance * np.outer(np.cos(theta), np.sin(phi)).flatten()
    y_circle = new_reduced_vector[1] + closest_distance * np.outer(np.sin(theta), np.sin(phi)).flatten()
    z_circle = new_reduced_vector[2] + closest_distance * np.outer(np.ones(100), np.cos(phi)).flatten()
    
    fig.add_trace(go.Scatter3d(x=x_circle, y=y_circle, z=z_circle, mode='markers', marker=dict(size=1, color='lightblue'), opacity=0.5))
    
    fig.show()
    
    return closest_title, closest_distance

# Mostrar el gráfico inicial
fig.show()

# Ejemplo de uso
new_text = input("Introduce un texto: ")
closest_title, closest_distance = add_and_visualize_new_text(new_text)
print(f"El título más cercano es: {closest_title} con una distancia de {closest_distance}")
