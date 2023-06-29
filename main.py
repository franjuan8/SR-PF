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

df_positivo = pd.read_csv('./positive.csv',encoding='utf-8')


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
