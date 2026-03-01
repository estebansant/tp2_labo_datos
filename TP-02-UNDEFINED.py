"""
Integrantes: Tobías Passarelli, Santiago Pizzani Esteban


"""
#%%

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

directorio_script = os.path.dirname(os.path.abspath(__file__))

data_ruta = os.path.join(directorio_script, "TP02-EnglishTypeAlphabet.csv")

df = pd.read_csv(data_ruta, low_memory=False)

#%%

# Verificar y explorar los datos
print(df.shape)
print(df.dtypes)

# Cantidad de valores nulos
print(df.isnull().sum())
print(df.head())
print(df['label'].value_counts().sort_index())

# El output en consola dice que cada uno de los labels tiene 1016 valores, ninguno de ellos es nulo, todos son tipo int64 


cuentas = df['label'].value_counts().sort_index()

# Imprimir el mínimo y máximo para el informe
print(f"Clase con más datos: {cuentas.max()}")
print(f"Clase con menos datos: {cuentas.min()}")

# Verificar nulos
nulos_totales = df.isnull().sum().sum()
print(f"Cantidad de valores nulos: {nulos_totales}")

# Rango de los pixeles
valor_min = df.iloc[:, 1:].values.min()
valor_max = df.iloc[:, 1:].values.max()
print(f"Rango de valores de píxeles: [{valor_min}, {valor_max}]")
print('es de 0 a 255')

# Chequeo de filas duplicadas
duplicados = df.duplicated().sum()
print(f"Cantidad de filas duplicadas: {duplicados}")

#%%

# Mapeo de label numérico a letra
label_a_letra = {}

for i in range(26):
    # NOTA: Como las letras mayusculas empiezan a partir del 65 en ASCII, empiezo a contar desde ese numero
    label_a_letra[i] = chr(65 + i)

fig, axes = plt.subplots(4, 7, figsize=(14, 8))
axes = axes.flatten()

for i in range(26):
    sample = df[df['label'] == i].iloc[0, 1:].values.reshape(28, 28)
    axes[i].imshow(sample, cmap='gray')
    axes[i].set_title(label_a_letra[i])
    axes[i].axis('off')

for ax in axes:
    ax.axis('off')
    
plt.tight_layout()
plt.show()

#%%

# Analisis estadistico por pixel

pixel_variance = df.iloc[:, 1:].var()

variance_image = pixel_variance.values.reshape(28, 28)
plt.imshow(variance_image, cmap='hot')
plt.colorbar()
plt.title('Varianza por pixel')
plt.xlabel('Píxeles en eje X')
plt.ylabel('Píxeles en eje Y')
plt.show()


#%%
# CODIGO DE TOBIAS

#1.a)
# Cantidad de datos y clases
print(f"Instancias: {df.shape[0]}, Atributos: {df.shape[1]}") # 
print(f"Letras únicas: {df['label'].nunique()}") #

# Separar lo que nos interesa del resto
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Separar los pixeles (X) de la letra (y)
X = df.iloc[:, 1:] 

# Calcular la varianza de cada pixel
varianzas = X.var()

# Rearmar la varianza como una imagen de 28x28
mapa_varianza = np.array(varianzas).reshape((28, 28))

# Graficar
plt.figure(figsize=(8, 6))
plt.imshow(mapa_varianza, cmap='hot')
plt.colorbar(label='Varianza')
plt.title("Relevancia de Atributos (Píxeles) según su Varianza")
plt.show()



# Varianzas de los 784 píxeles
varianzas = X.var()

# Umbral del 10%
umbral = varianzas.quantile(0.10) 
columnas_relevantes = varianzas[varianzas > umbral].index

# Filtrado del DF
X_reducido = X[columnas_relevantes]
print(f"Atributos originales: {X.shape[1]}")

print(f"Atributos descartados: {len(X.columns) - len(X_reducido.columns)}")



#%% 1.b)
# Revisado y con nuevos pares de letras

mapping = {
    12: 'M', 14: 'O', 16: 'Q', 18: 'S',
    15: 'P',  1: 'B',  3: 'D',  0: 'A',
    13: 'N', 10: 'K',  5: 'F'
}

def analizar_similitud(id1, id2, mapping):
    img1 = df[df['label'] == id1].iloc[:, 1:].mean().values.reshape(28, 28)
    img2 = df[df['label'] == id2].iloc[:, 1:].mean().values.reshape(28, 28)
    diff = np.abs(img1 - img2)
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img1, cmap='gray'); ax[0].set_title(f"Promedio {mapping[id1]}")
    ax[1].imshow(img2, cmap='gray'); ax[1].set_title(f"Promedio {mapping[id2]}")
    ax[2].imshow(diff, cmap='hot');  ax[2].set_title(f"Diferencia |{mapping[id1]}-{mapping[id2]}|")
    plt.tight_layout()
    plt.show()
    
    return np.linalg.norm(img1 - img2)

# Pares de letras
# O vs Q
dist_oq = analizar_similitud(14, 16, mapping)

# S vs M
dist_sm = analizar_similitud(18, 12, mapping)

# P vs B
dist_pb = analizar_similitud(15,  1, mapping)

# D vs O
dist_do = analizar_similitud( 3, 14, mapping)

# A vs N
dist_an = analizar_similitud( 0, 13, mapping)

# K vs F
dist_kf = analizar_similitud(10,  5, mapping)

# Grafico la distancia euclidea entre cada par
pares    = ['O vs Q', 'S vs M', 'P vs B', 'D vs O', 'A vs N', 'K vs F']
distancias = [dist_oq, dist_sm, dist_pb, dist_do, dist_an, dist_kf]

def ordenar(item):
    return item[1]

resumen = sorted(zip(pares, distancias), key=ordenar)

pares_ordenados = [x[0] for x in resumen]
distancias_ordenadas = [x[1] for x in resumen]

plt.figure(figsize=(10, 5))
bars = plt.bar(pares_ordenados, distancias_ordenadas)
plt.ylabel("Distancia Euclídea")

# Agregar el valor encima de cada barra
for bar, val in zip(bars, distancias_ordenadas):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{val:.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

print("\nRanking de similitud por distancia euclidea:")
for par, dist in resumen:
    print(f"  {par}: {dist:.2f}")

#%%1.c) 

df_j = df[df['label'] == 9]

# 16 imagenes al azar para ver la variedad de letras j
muestras_j = df_j.sample(16, random_state=42)

# Grilla de 4x4
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    # Lo hago que sea 28x28 y mantenga la forma
    img = muestras_j.iloc[i, 1:].values.reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.show()

# Ploteo el mapa de calor para las diferencias de J
subset = df[df['label'] == 9].iloc[:, 1:]
std_image = subset.std().values.reshape(28, 28)
plt.imshow(std_image, cmap='hot')
plt.colorbar()
plt.xlabel('Píxeles en eje X')
plt.ylabel('Píxeles en eje Y')
plt.show()

#%%

# Metricas para cuantificar al J
std_por_pixel = df_j.std()

print("Variabilidad de J")
print(f"Total de imágenes: {len(df_j)}")
print(f"Std promedio por píxel: {std_por_pixel.mean():.2f}")
print(f"Std máxima (píxel más variable): {std_por_pixel.max():.2f}")
print(f"Std mínima (píxel más estable): {std_por_pixel.min():.2f}")

# Agarro el promedio
j_promedio = df_j.mean().values

distancias = []
for i in range(len(df_j)):
    fila = df_j.iloc[i].values
    distancia = np.linalg.norm(fila - j_promedio)
    distancias.append(distancia)

distancias = np.array(distancias)


print(f"\nDistancia promedio a la J promedio: {distancias.mean():.2f}")
print(f"Distancia máxima a la J promedio:   {distancias.max():.2f}")
print(f"Distancia mínima a la J promedio:   {distancias.min():.2f}")

cantidad_bins = math.ceil(np.sqrt(len(distancias)))

# Histograma de distancias
plt.figure(figsize=(8, 4))
plt.hist(distancias, bins=cantidad_bins, color='steelblue', edgecolor='white')
plt.axvline(distancias.mean(), color='red', linestyle='--', label=f'Media: {distancias.mean():.1f}')
plt.xlabel("Distancia Euclídea a la J promedio")
plt.ylabel("Cantidad de imágenes")
plt.legend()
plt.tight_layout()
plt.show()

#%%

# DF con O y L

df_l = df[df['label'] == 11]
df_o = df[df['label'] == 14]
df_ol = pd.concat([df_l,df_o])

print(df_ol)