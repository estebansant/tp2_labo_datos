# -*- coding: utf-8 -*-
"""
Grupo: UNDEFINED
Integrantes: Tobías Passarelli, Santiago Pizzani Esteban

En este codigo analizamos el dataset con los caracteres del alfabeto ingles y le aplicamos un modelo de clasificaion binaria y otro de clasificacion multiple para diferenciar y clasificar las letras de las imagenes del dataset.
Lo primero que hacemos es explorar el dataset, ver la cantidad de datos que tiene, el tipo de datos, como estna organizados... Luego se ve la varianza para todas las letras y se grafica en un heattmap, se eligen apres de letras y se grafica la distancia euclidea entre ellas (el promedio de esa letra ideal), y por ultimo se usa el caso de la letra J como ejemplo para mostrar y analizar la diferencia que existe entre las distintas representaciones de una misma letra dentro del dataset.
Luego se busca aplicar un modelo de clasificacion binaria para las letras O y L. Para ello, primero se saca el promedio ideal de cada una de estas letras, se obtiene la varianza pixel a pixel y se ordena desde mayor varianza a menor varianza. Esto se usa para graficar la exactitud del modelo KNN en funcion de la cantidad de pixeles usados (y su orden en varianza). Se elige 50 como valor ideal y se pasa a graficar como cambia la exactitud para el KNN variando el K para el modelo con los primeros 50 pixeles de mayor varianza.
Finalmente se entreno un modelo de calsificacion multiclase usando un arbol de decision partiendo en 75% datos de desarrollo y 25% datos en heldout. Se vio primero un rango ideal de profundidades para evitar sobreajuste y subajuste, luego se probaron y compararon la exactitud de Gini contra la Entropia para ese rango de profundidades elegidas, de eso se obtuvo que 9 es la profundidad ideal. Por ultimo, para la profundidad 9 se probo el modelo usando el heldout con todos los datos de desarrollo como entrenamiento, se obtuvo la exactitud y se construyo la matriz de confusion.
"""
# %%

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

directorio_script = os.path.dirname(os.path.abspath(__file__))

data_ruta = os.path.join(directorio_script, "TP02-EnglishTypeAlphabet.csv")

df = pd.read_csv(data_ruta, low_memory=False)

# %%

# Verificar y explorar los datos
print(df.shape)
print(df.dtypes)


print(df.head())

# Cantidad de valores por label
print(df['label'].value_counts().sort_index())

# El output en consola dice que cada uno de los labels tiene 1016 valores, ninguno de ellos es nulo, todos son tipo int64


cuentas = df['label'].value_counts().sort_index()

# Imprimir el minimo y maximo para el informe
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

plt.savefig('graficos/heatmap_todas_las_letras.png',  dpi=300, bbox_inches='tight', transparent=False)
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

# Metemos el filtrado por etiqueta, el cálculo de la "letra promedio" con .mean(),  la generación de la imagen diferencia (np.abs) y el cálculo de la distancia euclidea en una misma funcion


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
    
    plt.savefig(f"graficos/paresLetras/par_{mapping[id1]}_{mapping[id2]}.png",  dpi=300, bbox_inches='tight', transparent=False)
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

# Valores en el centro y arriba de las barras
for bar, val in zip(bars, distancias_ordenadas):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + (max(distancias_ordenadas) * 0.01),
             f'{val:.0f}', 
             ha='center', 
             va='bottom', 
             fontsize=16)

plt.tight_layout()

plt.savefig('graficos/distancia_euclidiana_letras.png',  dpi=300, bbox_inches='tight', transparent=False)
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

plt.savefig('graficos/subplots_para_j.png',  dpi=300, bbox_inches='tight', transparent=False)
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
# Grafico de li­neas
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

plt.savefig('graficos/pixeles_para_knn.png',  dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# %% EJERCICIO 2.d: Variando el hiperparametro K
# Usaremos el mejor conjunto de atributos del top 50 pixeles con mayor varianza para buscar el mejor K

mejor_subconjunto = top_pixeles[:50]

# Probamos de 1 a 20 vecinos
k_range = range(1, 21)  
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

plt.savefig('graficos/variar_k_vecinos.png',  dpi=300, bbox_inches='tight', transparent=False)
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
print(f"Tamaños de los conjuntos (Total: {total})")
print(f"Desarrollo: {len(X_dev)} muestras ({len(X_dev)/total*100:.1f}%)")
print(
    f"Evaluación Final (Held-out): {len(X_held_out)} muestras ({len(X_held_out)/total*100:.1f}%)")
print(
    f"Dev-Train: {len(X_dev_train)} muestras ({len(X_dev_train)/total*100:.1f}%)")
print(
    f"Dev-Test: {len(X_dev_test)} muestras ({len(X_dev_test)/total*100:.1f}%)")

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

plt.savefig('graficos/profundidad_arbol.png',  dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Identificar la mejor profundidad
mejor_profundidad = depth_range[np.argmax(tree_test_precision)]
max_exactitud = max(tree_test_precision)

print(
    f"La mejor profundidad encontrada es: {mejor_profundidad} con una exactitud de {max_exactitud:.4f}")
# Profundidad 8
print("8:", tree_test_precision[7])
# Profundidad 9
print("9:", tree_test_precision[8])
# Profundidad 10
print("10:", tree_test_precision[9])
# Profundidad 12
print(tree_test_precision[11])
# Profundidad 14
print(tree_test_precision[13])
# Profundidad 19
print(tree_test_precision[18])

# Probablemente la mejor profundidad se encuentre entre 8 y el 12. Ya despues del 12 empieza a aparecer overfitting



#%% Punto 3.C)
# Seleccion de hiperparametros. Aqui vamos a comparar gini, entropia y profundidad del arbol

criterios = ['gini', 'entropy']

# Profundidad entre 1 y 10
profundidades = range(1, 11)

resultados = []

print("Inicio del kfolding")

# Exploramos todas las combinaciones posibles
for crit in criterios:
    print(f"va el criterio {crit}")
    for prof in profundidades:
        print(f"va en profundidad {prof}")
        arbol = DecisionTreeClassifier(criterion=crit, max_depth=prof, random_state=0)
        
        # Validación cruzada sobre el conjunto de desarrollo. EL cv divide la data 5 folds de forma que quede un 20% de datos de test para cada calculo de la exactitud
        exactitud = cross_val_score(arbol, X_dev, y_dev, cv=5)
        
        # Guardamos los resultados
        resultados.append({
            'Criterio': crit,
            'Profundidad': prof,
            'Exactitud_Media_CV': exactitud.mean()
        })
        #%%

# Pasar los resultados a un DF
df_resultados_hip = pd.DataFrame(resultados)

# Configuracion con la mejor exactitud
mejor_config = df_resultados_hip.loc[df_resultados_hip['Exactitud_Media_CV'].idxmax()]

print("Mejor configuracion encontrada")
print(f"Mejor criterio: {mejor_config['Criterio']}")
print(f"Profundidad optima: {mejor_config['Profundidad']}")
print(f"Mejor exactitud: {mejor_config['Exactitud_Media_CV'] * 100:.2f}%")


for crit in criterios:
    fila = df_resultados_hip[(df_resultados_hip['Criterio'] == crit) & (df_resultados_hip['Profundidad'] == 9)]
    exactitud = fila['Exactitud_Media_CV'].values[0]
    print(f"Criterio: {crit} | Profundidad: 9 | Exactitud Media CV: {exactitud*100:.2f}%")
    
#%% 
plt.figure(figsize=(10, 6))

# Funcion para hacer el grafico cada curva (una curva por criterio)
for criterio in df_resultados_hip['Criterio'].unique():
    # Filtro el DF para el criterio actual
    subset = df_resultados_hip[df_resultados_hip['Criterio'] == criterio]
    
    # Le cambio el nombre para que en la leyenda aparezcan como gini y entropia con mayuscula y bien escritos 
    nombre_label = "Gini" if "gini" in str(criterio).lower() else "Entropía"
    
    plt.plot(subset['Profundidad'], subset['Exactitud_Media_CV']*100, marker='o', label=nombre_label)

plt.xlabel('Profundidad Máxima del Árbol', fontsize=18)
plt.ylabel('Exactitud Media [%]', fontsize=18)
plt.xticks(range(1, 11), fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Criterio de Impureza', fontsize=16, title_fontsize=16)
plt.tight_layout()

plt.savefig('graficos/criterio_impureza.png',  dpi=300, bbox_inches='tight', transparent=False)
plt.show()
#%%
subset = df_resultados_hip[df_resultados_hip['Criterio'] == 'gini']
porcentajes = (subset['Exactitud_Media_CV']*100).round(2)
print(f"Exactitud media valores {porcentajes}")

# Decidimos elegir profundidad=9 porque la diferencia en exactitud con la profundidad 10 es de cerca de 1.2%, sin embargo el modelo se vuelve mas sencillo y rapido de ejecutar
# %% 
# 3.d) Evaluacion Held-out

# Entrenamiento con el mejor hiperparámetro (profundidad 9 y entropia) usando todos los datos del conjunto de desarrollo como entrenamiento del modelo
modelo_final = DecisionTreeClassifier(criterion='entropy', max_depth=9, random_state=0)
modelo_final.fit(X_dev, y_dev)

# Usamos el conjunto Held-out
exactitud_final = modelo_final.score(X_held_out, y_held_out)

# Crear una matriz de confusion para comparar que letras se confundieron mas con otras
y_pred = modelo_final.predict(X_held_out)
cm = confusion_matrix(y_held_out, y_pred)

print(f"Exactitud final en datos del held out: {exactitud_final * 100:.2f}%")

#%%
# Plotear la matriz de confusion
fig, ax = plt.subplots(figsize=(16, 13))
letras = [chr(i) for i in range(ord('A'), ord('Z') + 1)] # Genera etiquetas A-Z
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=letras)

# Grafico de mapa de colores con la matriz de confusion. A mayor intensidad de color, mas cantidad de apariciones de esa celda en la matriz
disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=True)
plt.setp(disp.text_, fontsize=14)

# Para agregar un label a la barra del color
cbar = disp.im_.colorbar
cbar.set_label('Cantidad de Apariciones', fontsize=18, labelpad=20)
cbar.ax.tick_params(labelsize=14)

plt.xlabel("Etiqueta Predicha por el Modelo", fontsize=18)
plt.ylabel("Etiqueta Real (Clasificada)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

plt.savefig('graficos/matriz_confusion.png',  dpi=300, bbox_inches='tight', transparent=False)
plt.show()

