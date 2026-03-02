"""
Integrantes: Tobías Passarelli, Santiago Pizzani Esteban


"""
# %%

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

directorio_script = os.path.dirname(os.path.abspath(__file__))

data_ruta = os.path.join(directorio_script, "TP02-EnglishTypeAlphabet.csv")

df = pd.read_csv(data_ruta, low_memory=False)

# %%

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

# Analisis estadistico por pixel

pixel_variance = df.iloc[:, 1:].var()

variance_image = pixel_variance.values.reshape(28, 28)
plt.imshow(variance_image, cmap='hot')
plt.colorbar()
plt.title('Varianza por pixel')
plt.xlabel('Píxeles en eje X')
plt.ylabel('Píxeles en eje Y')
plt.show()


# %%
# CODIGO DE TOBIAS

# 1.a)
# Cantidad de datos y clases
print(f"Instancias: {df.shape[0]}, Atributos: {df.shape[1]}")
print(f"Letras únicas: {df['label'].nunique()}")

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


# %% 1.b)
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
    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title(f"Promedio {mapping[id1]}")
    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title(f"Promedio {mapping[id2]}")
    ax[2].imshow(diff, cmap='hot')
    ax[2].set_title(f"Diferencia |{mapping[id1]}-{mapping[id2]}|")
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

# %%1.c)

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

# %%

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
plt.axvline(distancias.mean(), color='red', linestyle='--',
            label=f'Media: {distancias.mean():.1f}')
plt.xlabel("Distancia Euclídea a la J promedio")
plt.ylabel("Cantidad de imágenes")
plt.legend()
plt.tight_layout()
plt.show()

# %%

# DF con O y L

df_l = df[df['label'] == 11]
df_o = df[df['label'] == 14]
df_ol = pd.concat([df_l, df_o])

# Solo para verificar que le atine a los indices del label
print(df_ol)

print(df_ol.shape)
print(df_ol.dtypes)

# Cantidad de valores nulos
print(df_ol.isnull().sum())
print(df_ol.head())
print(df_ol['label'].value_counts().sort_index())

# El output en consola dice que cada uno de los labels tiene 1016 valores, ninguno de ellos es nulo, todos son tipo int64


cuentas = df_ol['label'].value_counts().sort_index()

# Imprimir el mínimo y máximo para el informe
print(f"Clase con más datos: {cuentas.max()}")
print(f"Clase con menos datos: {cuentas.min()}")

# Verificar nulos
nulos_totales = df_ol.isnull().sum().sum()
print(f"Cantidad de valores nulos: {nulos_totales}")

# Rango de los pixeles
valor_min = df_ol.iloc[:, 1:].values.min()
valor_max = df_ol.iloc[:, 1:].values.max()
print(f"Rango de valores de píxeles: [{valor_min}, {valor_max}]")
print('es de 0 a 255')

# Chequeo de filas duplicadas
duplicados = df_ol.duplicated().sum()
print(f"Cantidad de filas duplicadas: {duplicados}")

# %%

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

sub1 = top_pixeles[:3]

# Los 3 píxeles con MENOS varianza
sub2 = top_pixeles[-3:]

# 3 pixeles al azar (puedes elegir cualquier otro si quieres para probar tobias)
sub3 = top_pixeles[100:103]

subconjuntos = {
    'subconjunto1': sub1,
    'subconjunto2': sub2,
    'subconjunto3': sub3,
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

ax.set_ylabel('Exactitud')
ax.set_xticks(x)
ax.set_xticklabels(nombres, ha='right')
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.show()

# %%

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
    precision_train_mejores.append(accuracy_score(y_train, predicciones_train))

    # Probar modelo con test data
    predicciones_test = knn.predict(X_test[atributos])
    precision_test_mejores.append(accuracy_score(y_test, predicciones_test))

# Ahora repito para los pixeles de menor varianza

precision_train_peores = []
precision_test_peores = []

knn = KNeighborsClassifier(n_neighbors=3)

for nombre, atributos in subconjuntos_peor_varianza.items():
    # Construir modelo. De esta forma solo tengo los 3 pixeles seleccionados para armar mi modelo (cada modelo tiene 3 pixeles)
    knn.fit(X_train[atributos], y_train)

    # Precision en train data
    predicciones_train = knn.predict(X_train[atributos])
    precision_train_peores.append(accuracy_score(y_train, predicciones_train))

    # Probar modelo con test data
    predicciones_test = knn.predict(X_test[atributos])
    precision_test_peores.append(accuracy_score(y_test, predicciones_test))

cantidades = [3,10,50,100,200]
# Gráfico de Líneas
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
plt.ylabel('Exactitud', fontsize=18)
plt.xticks(cantidades, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %% EJERCICIO 2.d: Variando el hiperparámetro K
# Usaremos el mejor conjunto de atributos (ej: Top 50) para buscar el mejor K

mejor_subconjunto = top_pixeles[:50]
k_range = range(1, 21)  # Probamos de 1 a 20 vecinos
acc_train_k = []
acc_test_k = []

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train[mejor_subconjunto], y_train)

    acc_train_k.append(accuracy_score(
        y_train, knn_k.predict(X_train[mejor_subconjunto])))
    acc_test_k.append(accuracy_score(
        y_test, knn_k.predict(X_test[mejor_subconjunto])))

# Grafico de precision en funcion del K
plt.figure(figsize=(14, 8))
plt.plot(k_range, acc_train_k, label='Train Data',
         marker='o', linestyle='--', linewidth=3, markersize=10)
plt.plot(k_range, acc_test_k, label='Test Data', marker='s', linewidth=3, markersize=10)
plt.xlabel('Valor de K', fontsize=18)
plt.ylabel('Exactitud', fontsize=18)
plt.xticks(k_range, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Identificar el mejor K
mejor_k = k_range[np.argmax(acc_test_k)]
print(
    f"El mejor valor de K encontrado es: {mejor_k} con una exactitud de {max(acc_test_k):.4f}")

#%%

# Punto 3

#3a)


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

#%%

#Punto 3b)

tree_train_precision = []
tree_test_precision = []

depth_range = range(1, 21)

for i in range(1,21):
    tree = DecisionTreeClassifier(max_depth=i, random_state=0)
    tree.fit(X_dev_train, y_dev_train)
    tree_train_precision.append(tree.score(X_dev_train, y_dev_train))
    tree_test_precision.append(tree.score(X_dev_test, y_dev_test))

#%%
plt.figure(figsize=(16, 8))
plt.plot(depth_range, tree_train_precision, label='Train Data',
         marker='o', linestyle='--', linewidth=3, markersize=10)
plt.plot(depth_range, tree_test_precision, label='Test Data', marker='s', linewidth=3, markersize=10)
plt.xlabel('Profundidad del Árbol', fontsize=18)
plt.ylabel('Exactitud', fontsize=18)
plt.xticks(depth_range, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Identificar la mejor profundidad
mejor_profundidad = depth_range[np.argmax(tree_test_precision)]
max_exactitud = max(tree_test_precision)

print(f"La mejor profundidad encontrada es: {mejor_profundidad} con una exactitud de {max_exactitud:.4f}")
# Profundidad 14
print(tree_test_precision[13])
# Profundidad 19
print(tree_test_precision[18])

# Probablemente la mejor profundidad se encuentre entre 8 y el 12. Ya despues del 12 empieza a aparecer overfitting
