# Dog Breeds Embedding Visualization

Este proyecto visualiza embeddings basados en bigrams utilizando datos de razas de perros de The Dog API. La visualización permite ver las relaciones entre diferentes razas y cómo funciona un corrector ortográfico utilizando bigrams.

## Cómo funciona

1. **Generación de bigrams**: Transformar el término de búsqueda en bigrams para facilitar la comparación y búsqueda de patrones.
2. **Reducción de dimensionalidad**: Utilizar PCA (Análisis de Componentes Principales) para reducir los embeddings de bigrams a 3 dimensiones.
3. **Visualización interactiva**: Crear gráficos 3D interactivos utilizando Plotly para visualizar los embeddings.

## Ejecución del Proyecto

1. Clona este repositorio.
2. Instala las dependencias necesarias:
   ```sh
   pip install requests pandas numpy scikit-learn plotly scipy
3. Ejecuta el script 'embedding_visualization.py':
   ```sh
    python embedding_visualization.py

