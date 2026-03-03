# -*- coding: utf-8 -*-
"""
Integrantes: Tobías Passarelli, Santiago Pizzani Esteban


"""
# %%

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

directorio_script = os.path.dirname(os.path.abspath(__file__))

data_ruta = os.path.join(directorio_script, "letras.csv.gz")

df = pd.read_csv(data_ruta, low_memory=False)

# %%

# Verificar y explorar los datos
print(df.shape)
print(df.dtypes)

# Cantidad de valores nulos (SACARIA ESTO)
print(df.isnull().sum())


print(df.head())

# CANTIDAD DE VALORES POR LABEL
print(df['label'].value_counts().sort_index())

# El output en consola dice que cada uno de los labels tiene 1016 valores, ninguno de ellos es nulo, todos son tipo int64


cuentas = df['label'].value_counts().sort_index()

# Imprimir el mí­nimo y máximo para el informe
print(f"Clase con más datos: {cuentas.max()}")
print(f"Clase con menos datos: {cuentas.min()}")

# Verificar nulos (PODRIA IR CON LA VARIABLE DUPLICADOS)
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

# %%

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


# %%

# 1.a)
# Cantidad de datos y clases
print(f"Instancias: {df.shape[0]}, Atributos: {df.shape[1]}")
print(f"Letras Únicas: {df['label'].nunique()}")

# Separar lo que nos interesa del resto
y = df.iloc[:, 0]
# Analisis estadistico por pixel

# Separar los pixeles (X) de la letra (y)
X = df.iloc[:, 1:]

# Calcular la varianza de cada pixel
varianzas = X.var()

# Rearmar la varianza como una imagen de 28x28
mapa_varianza = np.array(varianzas).reshape((28, 28))

# Graficar
plt.figure(figsize=(10, 8))
img = plt.imshow(mapa_varianza, cmap='hot')

plt.xlabel("Columnas [Píxel]", fontsize=18)
plt.ylabel("Filas [Píxel]", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
cbar = plt.colorbar(img)
cbar.set_label('Varianza', fontsize=18)
cbar.ax.tick_params(labelsize=16)
plt.show()


# Umbral del 10%
umbral = varianzas.quantile(0.10)
columnas_relevantes = varianzas[varianzas > umbral].index

# Filtrado del DF
X_reducido = X[columnas_relevantes]
print(f"Atributos originales: {X.shape[1]}")

print(f"Atributos descartados: {len(X.columns) - len(X_reducido.columns)}")


# %% 1.b)
# Revisado y con nuevos pares de letras

# MAPEO DE NUMERO A LETRA
mapping = {
    12: 'M', 14: 'O', 16: 'Q', 18: 'S',
    15: 'P',  1: 'B',  3: 'D',  0: 'A',
    13: 'N', 10: 'K',  5: 'F'
}
"""
Encapsulamos el filtrado por etiqueta, el cálculo de la "letra promedio" mediante .mean(), 
la generación de la imagen diferencia (np.abs) y el cálculo de la distancia euclidea.
"""


def analizar_similitud(id1, id2, mapping):
    img1 = df[df['label'] == id1].iloc[:, 1:].mean().values.reshape(28, 28)
    img2 = df[df['label'] == id2].iloc[:, 1:].mean().values.reshape(28, 28)
    diff = np.abs(img1 - img2)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title(f"Promedio {mapping[id1]}", fontsize=18)
    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title(f"Promedio {mapping[id2]}", fontsize=18)
    ax[2].imshow(diff, cmap='hot')
    ax[2].set_title(f"Diferencia |{mapping[id1]}-{mapping[id2]}|", fontsize=18)
    
    fig.supxlabel("Columnas [Píxel]", fontsize=18)
    fig.supylabel("Filas [Píxel]", fontsize=18)
    
    for a in ax:
        a.tick_params(axis='both', labelsize=16)

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
dist_do = analizar_similitud(3, 14, mapping)

# A vs N
dist_an = analizar_similitud(0, 13, mapping)

# K vs F
dist_kf = analizar_similitud(10,  5, mapping)

# Grafico la distancia euclidea entre cada par
pares = ['O vs Q', 'S vs M', 'P vs B', 'D vs O', 'A vs N', 'K vs F']
distancias = [dist_oq, dist_sm, dist_pb, dist_do, dist_an, dist_kf]

# Función para ordenar los pares por su distancia


def ordenar(item):
    return item[1]


# Ordena los pares de menor a mayor distancia
resumen = sorted(zip(pares, distancias), key=ordenar)


pares_ordenados = [x[0] for x in resumen]
distancias_ordenadas = [x[1] for x in resumen]
#%%
plt.figure(figsize=(12, 6))
bars = plt.bar(pares_ordenados, distancias_ordenadas)

plt.ylabel("Distancia Euclidiana", fontsize=18)
plt.xlabel("Pares de Letras", fontsize=18)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(0, max(distancias_ordenadas) * 1.15)

for bar, val in zip(bars, distancias_ordenadas):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + (max(distancias_ordenadas) * 0.01),
             f'{val:.0f}', 
             ha='center', 
             va='bottom', 
             fontsize=16)

plt.tight_layout()
plt.show()

print("\nRanking de similitud por distancia euclidea:")
for par, dist in resumen:
    print(f"  {par}: {dist:.2f}")

# %% 1.c)

df_j = df[df['label'] == 9]

# 16 imagenes al azar para ver la variedad de letras j
muestras_j = df_j.sample(16, random_state=42)

# Metricas
std_por_pixel = df_j.iloc[:, 1:].std()
std_image = df[df['label'] == 9].iloc[:, 1:].std().values.reshape(28, 28)

j_promedio = df_j.iloc[:, 1:].mean().values
distancias = np.linalg.norm(df_j.iloc[:, 1:].values - j_promedio, axis=1)
cantidad_bins = math.ceil(np.sqrt(len(distancias)))

LABEL_FONTSIZE  = 16
TICK_FONTSIZE   = 16
LETTER_FONTSIZE = 18

fig = plt.figure(figsize=(16, 11))

# GridSpec usando 2 filas y 2 columnas
# Fila 0 tiene a los subplots (a) y (b)
# Fila 1 tiene al subplot  (c) ocupa ambas columnas
gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    hspace=0.20,
    wspace=0.10,
    height_ratios=[1.1, 0.9]
)

# Grilla 4×4 de J
ax_grid_outer = fig.add_subplot(gs[0, 0])
ax_grid_outer.axis('off')
ax_grid_outer.text(
    -0.08, 0.96, '(a)',
    transform=ax_grid_outer.transAxes,
    fontsize=18, fontweight='bold', va='top'
)

inner_gs = gridspec.GridSpecFromSubplotSpec(
    4, 4, subplot_spec=gs[0, 0], hspace=0.05, wspace=0.05
)
for i in range(16):
    ax_img = fig.add_subplot(inner_gs[i])
    img = muestras_j.iloc[i, 1:].values.reshape(28, 28)
    ax_img.imshow(img, cmap='gray')
    ax_img.axis('off')

#Mapa de calor de desviación estándar
ax_heat = fig.add_subplot(gs[0, 1])
img = ax_heat.imshow(std_image, cmap='hot')
cbar = fig.colorbar(img, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.set_label('Varianza', fontsize=18)
cbar.ax.tick_params(labelsize=16)

ax_heat.set_xlabel('Columnas [Píxel]', fontsize=18)
ax_heat.set_ylabel('Filas [Píxel]', fontsize=18)
ax_heat.tick_params(axis='both', labelsize=16)
ax_heat.text(
    -0.24,  0.96, '(b)',
    transform=ax_heat.transAxes,
    fontsize=18, fontweight='bold', va='top'
)

# Histograma de distancias euclidianas

# Esto hace que el plot ocupe las 2 columnas enteras
ax_hist = fig.add_subplot(gs[1, :])
ax_hist.hist(distancias, bins=cantidad_bins, color='steelblue', edgecolor='white')
ax_hist.axvline(
    distancias.mean(), color='red', linestyle='--',
    label=f'Media: {distancias.mean():.1f}'
)
ax_hist.set_xlabel('Distancia Euclidiana a la J promedio', fontsize=18)
ax_hist.set_ylabel('Cantidad de Imágenes',                 fontsize=18)
ax_hist.tick_params(axis='both', labelsize=16)
ax_hist.legend(fontsize=16)
ax_hist.text(
    0.02, 0.94, '(c)',
    transform=ax_hist.transAxes,
    fontsize=18, fontweight='bold', va='top'
)

plt.savefig('subplots_j.png', dpi=300, bbox_inches='tight')
plt.show()
# %% 2.a)

# DF con O y L

clases_interes = [11, 14]  # L y O
df_ol = df[df['label'].isin(clases_interes)].copy()

# Solo para verificar que le atine a los indices del label
print(df_ol)

print(df_ol.shape)
print(df_ol.dtypes)

# Cantidad de valores nulos
print(df_ol.isnull().sum())

print(df_ol['label'].value_counts().sort_index())

# El output en consola dice que cada uno de los labels tiene 1016 valores, ninguno de ellos es nulo, todos son tipo int64


cuentas = df_ol['label'].value_counts().sort_index()

# Imprimir el mi­nimo y maximo para el informe
print(f"Clase con más datos: {cuentas.max()}")
print(f"Clase con menos datos: {cuentas.min()}")

# Verificar nulos
nulos_totales = df_ol.isnull().sum().sum()
print(f"Cantidad de valores nulos: {nulos_totales}")

# Rango de los pixeles
valor_min = df_ol.iloc[:, 1:].values.min()
valor_max = df_ol.iloc[:, 1:].values.max()
print(f"Rango de valores de pixeles: [{valor_min}, {valor_max}]")
print('es de 0 a 255')

# Chequeo de filas duplicadas
duplicados = df_ol.duplicated().sum()
print(f"Cantidad de filas duplicadas: {duplicados}")

# %% 2.b)
# En el informe: No se realizó escalado de atributos
# dado que todos los píxeles comparten la misma unidad y rango dinámico ([0,255])


# Separo en datos de train y test

pixeles = df_ol.drop('label', axis=1)
letra = df_ol['label']

# El random_state=66 sale del libro jajaja. Es solo el orden con el que se ordenan de forma aleatoria los valores del df y se asegura que se mantenga en ese orden al volver a correr el codigo (puede ser cualquier otro numero)
X_train, X_test, y_train, y_test = train_test_split(
    pixeles, letra, test_size=0.25, random_state=66, shuffle=True, stratify=letra)

print(f"Total de datos originales: {len(df_ol)}")
print(f"Datos para entrenamiento (X_train): {len(X_train)} filas")
print(f"Datos para prueba (X_test): {len(X_test)} filas")

varianza_ol = X_train.var().sort_values(ascending=False)

# Los indices de los pixeles que mas varian
top_pixeles = varianza_ol.index.tolist()

# solo para ver. podemos usar estos e ir de 3 en 3 para probar distintas variables y ver si cambia mucho o no tanto
print("10 pixeles más variables:")
print(top_pixeles[:10])
# En el informe: No se realizó escalado de atributos
# dado que todos los píxeles comparten la misma unidad y rango dinámico ([0,255])

sub1 = top_pixeles[:3]

# Los 3 píxeles con MENOS varianza
sub2 = top_pixeles[-3:]

# 3 pixeles al azar
sub3 = top_pixeles[100:103]

subconjuntos = {
    'Top 3 Varianza': sub1,
    'Bottom 3 Varianza': sub2,
    '3 Varianza Media': sub3,
}


knn = KNeighborsClassifier(n_neighbors=3)

precision_train = []
precision_test = []


for nombre, atributos in subconjuntos.items():
    # Construir modelo. De esta forma solo tengo los 3 pixeles seleccionados para armar mi modelo (cada modelo tiene 3 pixeles)
    knn.fit(X_train[atributos], y_train)

    # Precision en train data
    predicciones_train = knn.predict(X_train[atributos])
    precision_train.append(accuracy_score(y_train, predicciones_train))

    # Probar modelo con test data
    predicciones_test = knn.predict(X_test[atributos])
    precision_test.append(accuracy_score(y_test, predicciones_test))


# Plot para el grafico con data de precision.
nombres = list(subconjuntos.keys())
x = np.arange(len(nombres))
ancho = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
barras_train = ax.bar(x - ancho/2, precision_train, ancho,
                      label='Train')
barras_test = ax.bar(x + ancho/2, precision_test,  ancho,
                     label='Test')

# Valor encima de cada barra
for barra in barras_train:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.005,
            f'{barra.get_height():.3f}', ha='center', va='bottom', fontsize=9)

for barra in barras_test:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.005,
            f'{barra.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Exactitud [%]')
ax.set_xticks(x)
ax.set_xticklabels(nombres, ha='right')
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.show()

# En el informe: No se realizó escalado de atributos
# dado que todos los píxeles comparten la misma unidad y rango dinámico ([0,255])

# ¿cuáles son los píxeles físicamente?
# Imprimimos sus coordenadas (fila, columna)
for nombre, atributos in subconjuntos.items():
    print(f"\n{nombre}:")
    for attr in atributos:
        pixel_idx = int(attr.replace('pixel', ''))
        fila = pixel_idx // 28
        col = pixel_idx % 28
        print(f"  - {attr}: [Fila {fila}, Col {col}]")

# %% 2.c)

# Ahora quiero ver como cambia la precision para distintas cantidades de atributos seleccionados de varias partes de los pixeles (mayor y menor varianza)
subconjuntos_mejor_varianza = {
    '3 Pixeles (Comunes)': top_pixeles[:3],
    '10 Pixeles (Comunes)': top_pixeles[:10],
    '50 Pixeles (Comunes)': top_pixeles[:50],
    '100 Pixeles (Comunes)': top_pixeles[:100],
    '200 Pixeles (Comunes)': top_pixeles[:200],
}

subconjuntos_peor_varianza = {
    '3 Pixeles (Menos Comunes)': top_pixeles[-3:],
    '10 Pixeles (Menos Comunes)': top_pixeles[-10:],
    '50 Pixeles (Menos Comunes)': top_pixeles[-50:],
    '100 Pixeles (Menos Comunes)': top_pixeles[-100:],
    '200 Pixeles (Menos Comunes)': top_pixeles[-200:],
}

# Para los pixeles con mayor varianza

precision_train_mejores = []
precision_test_mejores = []

knn = KNeighborsClassifier(n_neighbors=3)

for nombre, atributos in subconjuntos_mejor_varianza.items():
    # Construir modelo. De esta forma solo tengo los 3 pixeles seleccionados para armar mi modelo (cada modelo tiene 3 pixeles)
    knn.fit(X_train[atributos], y_train)

    # Precision en train data
    predicciones_train = knn.predict(X_train[atributos])
    precision_train_mejores.append(
        accuracy_score(y_train, predicciones_train)*100)

    # Probar modelo con test data
    predicciones_test = knn.predict(X_test[atributos])
    precision_test_mejores.append(
        accuracy_score(y_test, predicciones_test)*100)

# Ahora repito para los pixeles de menor varianza

precision_train_peores = []
precision_test_peores = []

knn = KNeighborsClassifier(n_neighbors=3)

for nombre, atributos in subconjuntos_peor_varianza.items():
    # Construir modelo. De esta forma solo tengo los 3 pixeles seleccionados para armar mi modelo (cada modelo tiene 3 pixeles)
    knn.fit(X_train[atributos], y_train)

    # Precision en train data
    predicciones_train = knn.predict(X_train[atributos])
    precision_train_peores.append(
        accuracy_score(y_train, predicciones_train)*100)

    # Probar modelo con test data
    predicciones_test = knn.predict(X_test[atributos])
    precision_test_peores.append(accuracy_score(y_test, predicciones_test)*100)

cantidades = [3, 10, 50, 100, 200]
# GrÃ¡fico de LÃ­neas
plt.figure(figsize=(14, 8))
plt.plot(cantidades, precision_train_mejores, marker='o', linewidth=3, markersize=10,
         label='Train Data | Píxeles con Mayor Varianza', color='#1F77B4', linestyle='--')
plt.plot(cantidades, precision_test_mejores, marker='s', linewidth=3, markersize=10,
         label='Test Data | Píxeles con Mayor Varianza', color='#1F77B4')
plt.plot(cantidades, precision_test_peores, marker='s', linewidth=3, markersize=10,
         label='Test Data | Píxeles Menor Varianza', color='#FF7F0E')
plt.plot(cantidades, precision_train_peores, marker='o', linewidth=3, markersize=10,
         label='Train Data | Píxeles Menor Varianza', color='#FF7F0E', linestyle='--')

plt.xlabel('Cantidad de Píxeles Utilizados', fontsize=18)
plt.ylabel('Exactitud [%]', fontsize=18)
plt.xticks(cantidades, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %% EJERCICIO 2.d: Variando el hiperparametro K
# Usaremos el mejor conjunto de atributos (ej: Top 50) para buscar el mejor K

mejor_subconjunto = top_pixeles[:50]
k_range = range(1, 21)  # Probamos de 1 a 20 vecinos
acc_train_k = []
acc_test_k = []

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train[mejor_subconjunto], y_train)

    acc_train_k.append(accuracy_score(
        y_train, knn_k.predict(X_train[mejor_subconjunto])) * 100)
    acc_test_k.append(accuracy_score(
        y_test, knn_k.predict(X_test[mejor_subconjunto]))*100)

# Grafico de precision en funcion del K
plt.figure(figsize=(14, 8))
plt.plot(k_range, acc_train_k, label='Train Data (%)',
         marker='o', linestyle='--', linewidth=3, markersize=10)
plt.plot(k_range, acc_test_k, label='Test Data (%)',
         marker='s', linewidth=3, markersize=10)
plt.xlabel('Valor de K', fontsize=18)
plt.ylabel('Exactitud [%]', fontsize=18)
plt.xticks(k_range, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Identificar el mejor K
mejor_k = k_range[np.argmax(acc_test_k)]
print(
    f"El mejor valor de K encontrado es: {mejor_k} con una exactitud de {max(acc_test_k):.4f}")

# %%

# Punto 3

# 3a)


pixeles = df.drop('label', axis=1)
letra = df['label']
# Separo en held_out y dev
X_dev, X_held_out, y_dev, y_held_out = train_test_split(
    pixeles, letra, test_size=0.25, random_state=66, shuffle=True, stratify=letra)

# Separo la data de dev en 80% train y 20% test
X_dev_train, X_dev_test, y_dev_train, y_dev_test = train_test_split(
    X_dev, y_dev,
    test_size=0.20,
    random_state=66,
    stratify=y_dev
)

# Verificacion de tamaños para el informe
total = len(df)
print(f" Tamaños de los conjuntos (Total: {total})")
print('---')
print(f"Desarrollo (Dev): {len(X_dev)} muestras ({len(X_dev)/total*100:.1f}%)")
print(
    f"Evaluación Final (Held-out): {len(X_held_out)} muestras ({len(X_held_out)/total*100:.1f}%)")
print(
    f"  └─ Dev-Train: {len(X_dev_train)} muestras ({len(X_dev_train)/total*100:.1f}%)")
print(
    f"  └─ Dev-Test (Val): {len(X_dev_test)} muestras ({len(X_dev_test)/total*100:.1f}%)")

# %%

# Punto 3b)

tree_train_precision = []
tree_test_precision = []

depth_range = range(1, 21)

for i in depth_range:
    tree = DecisionTreeClassifier(max_depth=i, random_state=0)
    tree.fit(X_dev_train, y_dev_train)
    tree_train_precision.append(tree.score(X_dev_train, y_dev_train)*100)
    tree_test_precision.append(tree.score(X_dev_test, y_dev_test)*100)

# %%

plt.figure(figsize=(16, 6))
plt.plot(depth_range, tree_train_precision, label='Train Data',
         marker='o', linestyle='--', linewidth=3, markersize=10)
plt.plot(depth_range, tree_test_precision, label='Test Data',
         marker='s', linewidth=3, markersize=10)
plt.xlabel('Profundidad del Árbol', fontsize=18)
plt.ylabel('Exactitud [%]', fontsize=18)
plt.xticks(depth_range, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Identificar la mejor profundidad
mejor_profundidad = depth_range[np.argmax(tree_test_precision)]
max_exactitud = max(tree_test_precision)

print(
    f"La mejor profundidad encontrada es: {mejor_profundidad} con una exactitud de {max_exactitud:.4f}")
# Profundidad 8
print(tree_test_precision[9])
# Profundidad 12
print(tree_test_precision[11])
# Profundidad 14
print(tree_test_precision[13])
# Profundidad 19
print(tree_test_precision[18])

# Probablemente la mejor profundidad se encuentre entre 8 y el 12. Ya despues del 12 empieza a aparecer overfitting


# %% Punto 3.c)

# Experimento de Selección con K-Folding

# Definimos el rango limitado entre 1 y 10 de profundidad
depth_range_consigna = range(1, 11)
promedios_cv = []

print("Iniciando validación cruzada (K-Folding)...")

for d in depth_range_consigna:
    model = DecisionTreeClassifier(max_depth=d, random_state=0)

    # Usamos cross_val_score sobre todo el conjunto X_dev
    scores = cross_val_score(model, X_dev, y_dev, cv=5)
    promedios_cv.append(scores.mean())
    print(f"Profundidad {d}: Exactitud promedio CV = {scores.mean():.4f}")

# Seleccionamos el mejor modelo según la consigna
mejor_prof_cv = depth_range_consigna[np.argmax(promedios_cv)]
mejor_score_cv = max(promedios_cv)

print(f"La mejor configuración es max_depth = {mejor_prof_cv}")
print(f"Performance (Exactitud promedio CV): {mejor_score_cv * 100:.2f}%")


# Gráfico de validación cruzada
plt.figure(figsize=(10, 6))
plt.plot(depth_range_consigna, promedios_cv, marker='o',
         linestyle='-', color='forestgreen', linewidth=2)
plt.xlabel("Profundidad Máxima (max_depth)")
plt.ylabel("Exactitud Promedio (Mean CV Accuracy)")
plt.xticks(depth_range_consigna)
plt.grid(True, alpha=0.3)
# Resaltamos el punto óptimo
plt.axvline(mejor_prof_cv, color='red', linestyle='--',
            label=f'Mayor Exactitud : Profundidad = {mejor_prof_cv}')
plt.legend()
plt.show()


# %% Punto 3.d)
# Evaluación final del modelo SELECCIONADO (Depth 10)
"""
modelo_seleccionado = DecisionTreeClassifier(max_depth=10, random_state=0)
modelo_seleccionado.fit(X_dev, y_dev)  # Re-entrenamos con todo Dev

exactitud_final_legal = modelo_seleccionado.score(X_held_out, y_held_out)

print(
    f"Exactitud final en Held-out (Modelo Depth 10): {exactitud_final_legal * 100:.2f}%")
"""
# hay que meterle metricas de test a esto. Exactitud y Matriz de Confusion

# Punto 3.d) Evaluación Final: Exactitud y Matriz de Confusión

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Punto 3.d) Evaluación Final sobre Held-out

# 1. Entrenamiento final con el hiperparámetro seleccionado (depth=10)
# Se utiliza el conjunto de Desarrollo completo (X_dev, y_dev) [cite: 311, 350]
modelo_final = DecisionTreeClassifier(max_depth=10, random_state=0)
modelo_final.fit(X_dev, y_dev)

# 2. Evaluación ÚNICA sobre el conjunto Held-out [cite: 325]
exactitud_final = modelo_final.score(X_held_out, y_held_out)

# 3. Matriz de Confusión para análisis de errores [cite: 677, 1037]
y_pred = modelo_final.predict(X_held_out)
cm = confusion_matrix(y_held_out, y_pred)

print("--- RESULTADO DEFINITIVO ---")
print(f"Exactitud final en datos no vistos (Held-out): {exactitud_final * 100:.2f}%")

# (Aquí va el código del plot de la matriz que ya tenés)

# Configuración visual para el informe
fig, ax = plt.subplots(figsize=(14, 11))
letras = [chr(i) for i in range(ord('A'), ord('Z') + 1)] # Genera etiquetas A-Z
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=letras)

# Graficamos con mapa de colores Blues para facilitar la lectura de la diagonal [cite: 1054]
disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=True)

plt.title(f"Matriz de Confusión Final - Exactitud: {exactitud_final*100:.2f}%", fontsize=14)
plt.xlabel("Etiqueta Predicha por el Modelo", fontsize=12)
plt.ylabel("Etiqueta Real (Datos Held-out)", fontsize=12)
plt.tight_layout()
plt.show()

