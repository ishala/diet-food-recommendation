from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import pickle
import os
import random
import logging
logging.basicConfig(level=logging.DEBUG)

def modelling():
    # Ambil dataset
    df = pd.read_csv('static/data/cleaned_diets.csv')
    # Inisiasi TFIDF
    tfidf = TfidfVectorizer()
    
    # Pengambilan nilai unique dari data
    uniqueVal = df['fitur_tfidf'].unique()
    
    # Ubah dalam bentuk matrix
    tfidfMatrix = tfidf.fit_transform(uniqueVal)
    
    # Simpan model TFIDF
    with open('models/tfidf_model.pkl', 'wb') as file:
        pickle.dump(tfidf, file)
    
    # Simpan tfidfMatrix untuk digunakan nanti
    with open('models/tfidf_matrix.pkl', 'wb') as file:
        pickle.dump(tfidfMatrix, file)
    
    return df, tfidf, tfidfMatrix

def defaultLevel(formData):
    listLevel = [0, 1, 2, 3]
    
    if not formData.get('protein'):
        formData['protein'] = random.choice(listLevel)
    if not formData.get('carbs'):
        formData['carbs'] = random.choice(listLevel)
    if not formData.get('fat'):
        formData['fat'] = random.choice(listLevel)
    
    return formData


def getRecommendations(formData, n=10):
    if not (os.path.exists('models/tfidf_model.pkl') and os.path.exists('models/tfidf_matrix.pkl')):
        df, tfidf, tfidfMatrix = modelling()
    else:
        df = pd.read_csv('static/data/cleaned_diets.csv')
        with open('models/tfidf_model.pkl', 'rb') as file:
            tfidf = pickle.load(file)
        with open('models/tfidf_matrix.pkl', 'rb') as file:
            tfidfMatrix = pickle.load(file)
            
    logging.debug(f"Input formData: {formData}")
    # Copy data asli untuk standarisasi
    dfCopy = df.copy()
        
    # Standarisasi data nutrisi
    # scaler = StandardScaler()
    # dfCopy[['kadar_protein', 'kadar_karbo', 'kadar_lemak']] = scaler.fit_transform(dfCopy[['kadar_protein', 'kadar_karbo', 'kadar_lemak']])
    
    # buat numpy di nutritionalData
    nutritionalData = np.array([float(formData['protein']), float(formData['carbs']), float(formData['fat'])])
    # nutritionalData = scaler.transform(nutritionalData.reshape(1, -1))
    
    # gabung inputan tipe diet dan makanan
    inputText = f"{formData['diet-type']} {formData['items']}"
    # Transformasi jadi vektor tfidf
    inputTfidf = tfidf.transform([inputText])
    
    # pencarian nilai similarity
    simScores = cosine_similarity(inputTfidf, tfidfMatrix).flatten()
    simScores = (simScores - simScores.min()) / (simScores.max() - simScores.min())
    
    # Reshape nutritional data
    nutritionalDfReshaped = nutritionalData.reshape(1, -1)
    
    # Menambahkan nilai similarity nutritionalData
    nutritionalSim = cosine_similarity(nutritionalDfReshaped, dfCopy[['segmentasi_kadar_protein', 'segmentasi_kadar_karbo', 'segmentasi_kadar_lemak']].values).flatten()
    nutritionalSim = (nutritionalSim - nutritionalSim.min()) / (nutritionalSim.max() - nutritionalSim.min())
    
    # Kombinasi kedua nilai similarity dengan euclidean distance
    alpha = 0.3
    combinedSim = (alpha * simScores) + ((1 - alpha) * nutritionalSim)
    
    # Mengurutkan skor similaritas dari tinggi ke rendah
    sortedScores = combinedSim.argsort()[::-1]
    
    # Mengambil top N nilai indeks similar
    topNIndex = sortedScores[:n]
    
    # Mengambil data dari data asli dengan indeks yang didapatkan
    topNRecomended = df.iloc[topNIndex].copy()
    
    # Menghapus kolom fitur_tfidf
    topNRecomended = topNRecomended.drop(['fitur_tfidf', 'segmentasi_kadar_protein', 'segmentasi_kadar_karbo','segmentasi_kadar_lemak'], axis=1)
    
    recomendedDict = topNRecomended.to_dict(orient='records')
    return recomendedDict
