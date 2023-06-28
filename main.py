import pandas as pd
from fastapi import FastAPI
import ast
import locale
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import randomized_svd
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title='Proyecto Final',
              description='Data 10'
              )

@app.get('/')
async def read_root():
    return{"Mi API para el proyecto final"}

@app.get('/')
async def index():
    return {"API hecha por Juan Alfaro"}

@app.get('/about/')
async def about():
    return {'Proyecto final de la cohorte 10 acerca de un sistema de recomendacion'}


df = pd.read_csv('./reviews2.csv', encoding='utf-8')
df_positivo = pd.read_csv('./positive.csv',encoding='utf-8')
# El Vectorizador TfidfVectorizer con parámetros de reduccion procesamiento
df['category'].fillna('', inplace=True)
vectorizar = TfidfVectorizer(min_df=10, max_df=0.5, ngram_range=(1,2))

# Vectorizamos, ajustamos y transformamos el texto de la columna "title" del DataFrame
X = vectorizar.fit_transform(df['category'])

# Calcular la matriz de similitud de coseno con una matriz reducida de 7000
similarity_matrix = cosine_similarity(X[:1250,:])

# Obtenemos la descomposición en valores singulares aleatoria de la matriz de similitud de coseno con 10 componentes
n_components = 10
U, Sigma, VT = randomized_svd(similarity_matrix, n_components=n_components)

# Construir la matriz reducida de similitud de coseno
reduced_similarity_matrix = U.dot(np.diag(Sigma)).dot(VT)

@app.get('/recomendacionPositives/{id_unic}')
def recomendacion2(id:int):
    '''Ingresas un id y te recomienda las similares en una lista'''
    visited_id = id

    if (df_positivo['id_unic'] == visited_id).any():
        visited_local = df_positivo[df_positivo['id_unic'] == visited_id]['local_name'].iloc[0]
        visited_ids = df_positivo[df_positivo['local_name'] == visited_local]['id_unic'].unique()

        df_recomendaciones = df_positivo.loc[
            (df_positivo['category'].str.lower().str.contains('restaurant')) & 
            (~df_positivo['id_unic'].isin(visited_ids))
        ]

    recomendaciones = df_recomendaciones[~df_recomendaciones['local_name'].duplicated(keep='first')]['local_name'].tolist()
    recomendaciones = recomendaciones[:6]  # Limitar a un máximo de 6 recomendaciones
    
    return recomendaciones


@app.get('/recomendacion/{user}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de un local y te recomienda las similares en una lista'''
    titulo = titulo.title()
    
    indice = np.where(df['local_name'] == titulo)[0][0]
     
    puntuaciones_similitud = reduced_similarity_matrix[indice,:]
    
    puntuacion_ordenada = np.argsort(puntuaciones_similitud)[::-1]

    top_indices = puntuacion_ordenada[:5]
    
    return df.loc[top_indices, 'local_name'].tolist()