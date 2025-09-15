#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


# In[2]:


# Carga y tratamiento del dataset
 # ------------ Carga -----------------
data_tr=pd.read_csv('afluencia_desglosada_trolebus_05_2025.csv')

 # Normaliza el nombre de las líneas en la columna linea
data_tr['linea'] = data_tr['linea'].str.lower().str.replace('línea', 'linea', regex=False).str.strip()

 # Verifica de que la columna 'fecha' esté en formato de fecha
data_tr['fecha'] = pd.to_datetime(data_tr['fecha'])

 # Ahora, crea la nueva columna 'dia_semana' extrayendo el nombre del día
data_tr['dia_semana'] = data_tr['fecha'].dt.day_name()


# In[3]:


# Categorización de la afluencia en baja, media o alta por el método de cuartiles
#----------------------------------------------
  # Creación de los cuartiles
c1=data_tr['afluencia'].quantile(0.25)
c3=data_tr['afluencia'].quantile(0.75)
#----------------------------------------------
  # función de clasificación
def clasificador_afluencia(afluencia):
    if afluencia <=c1:
        return 'Baja'
    elif afluencia <= c3:
        return 'Media'
    else:
        return 'Alta'
#----------------------------------------------
  # Se crea una nueva columna con la categorización y se agrega al dataset
data_tr['categoria_afluencia'] = data_tr['afluencia'].apply(clasificador_afluencia)


# In[4]:


# Visualiza la información del dataset
data_tr.info()


# In[5]:


# Se plantea el modelo para luego analizar la importancia de las variables
X = data_tr[['anio','mes','linea','dia_semana']]
y = data_tr['categoria_afluencia']

# --- Aplica One-Hot Encoding a la variable 'mes' que está en X. ---
X_preprocesado = pd.get_dummies(X, columns=['mes','dia_semana','linea'], drop_first=True)

# --- Convierte la variable objetivo 'y' a números (Baja=0, Media=1, Alta=2) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Separar los datos en entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X_preprocesado, y_encoded, test_size=0.2, random_state=42)

# --- Entrenamiento del modelo Random Forest --- 
modelo_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_random_forest.fit(X_train, y_train)

# --- Paso 4: Evaluar el modelo y hacer una predicción ---
y_pred = modelo_random_forest.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo para este tipo de transporte: {precision:.2f}\n")


# In[6]:


# Analizando la importancia de las variables (selección de variables) para quedarnos con aquellas que hagan más preciso al modelo
importancias = modelo_random_forest.feature_importances_
nombres_variables = X_preprocesado.columns

importancia_df = pd.DataFrame({
    'variable': nombres_variables,
    'importancia': importancias
}).sort_values(by='importancia', ascending=False)

print("Importancia de las variables:")
print(importancia_df)


# Se aprecia que los valores con más importancia se centran en el año y la línea. Por esta razón, se replanteará el modelo con estas variables y se agregará la varible mes para tener una predicción más completa.

# In[7]:


# Se replantea el modelo con base en el análisis de la importacia de variables 
X = data_tr[['anio', 'linea','mes']]
y = data_tr['categoria_afluencia']

# --- Aplica One-Hot Encoding a la variable 'linea' ---
X_preprocesado = pd.get_dummies(X, columns=['linea','mes'], drop_first=True)

# --- Convertierte la variable objetivo 'y' a números (Baja=0, Media=1, Alta=2) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Dividisión de los datos en entrenamiento y prueba ---
X_train, X_test, y_train, y_test = train_test_split(X_preprocesado, y_encoded, test_size=0.2, random_state=42)

# Entrenamiento y aplicación de SMOTE para que todas las categorías aparezcan en la misma proporción
# --- Entrena y evalúa el modelo SIN SMOTE ---
# ----- El modelo aprende con los datos de entrenamiento originales y desbalanceados -----
modelo_sin_smote = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_sin_smote.fit(X_train, y_train)

y_pred_sin_smote = modelo_sin_smote.predict(X_test)
precision_sin_smote = accuracy_score(y_test, y_pred_sin_smote)

# --- Aplicaa SMOTE y entrena el modelo FINAL ---
# ----- Aplica SMOTE para equilibrar los datos de entrenamiento -----
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ----- Entrena el modelo final con los datos balanceados -----
modelo_final = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_final.fit(X_train_smote, y_train_smote)

# ----- Evaluar el modelo con los datos de prueba originales -----
y_pred_final = modelo_final.predict(X_test)
precision_final = accuracy_score(y_test, y_pred_final)

# Muestra la distribución original antes de SMOTE (véase desbalance de los datos)
print("Distribución de la variable objetivo en los datos de entrenamiento originales:")
print(pd.Series(y_train).value_counts())
print("\n")

# Muestra la predicción sin SMOTE
print(f"Precisión del modelo SIN SMOTE: {precision_sin_smote:.2f}")

print("-----------------------------------------------------------\n")

# Muestra la distribución después de SMOTE
print("Distribución de la variable objetivo después de SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Muestra la predicción sin SMOTE
print(f"\nPrecisión del modelo CON SMOTE: {precision_final:.2f}")

print("-----------------------------------------------------------\n")

# Para una evaluación más completa...
print("\nInforme de clasificación del modelo CON SMOTE:\n")
print(classification_report(y_test, y_pred_final, target_names=label_encoder.classes_))


# In[8]:


print(data_tr['categoria_afluencia'].value_counts())


# Note que el modelo pierde presición al aplicar SMOTE pero, ahora las tres categorías de la variable objetivo 'categoria_afluencia' son entrenadas con el mismo número de datos. Esto mejora el modelo porque impide que éste asigne la categoría 'Media' (que es la de mayor frecuencia) al alcanzar el 50%; si esto sucediera estaría adivinando la afluencia en base al valor con mayor repitencia (sería un modelo "perezoso").
# 
# cita: <i><b>SMOTE</b> (Synthetic Minority Over-sampling Technique) es una técnica que se usa para resolver un problema común en los datos: el desbalance de clases</i>.

# In[9]:


# PREDICCIÓN 
# --- Crea los datos de 2025 por mes ---
lineas_transporte = sorted(data_tr['linea'].unique())
meses_del_2025 = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
anio_2025 = 2025

# Creaa un DataFrame vacío para las predicciones
datos_a_predecir = pd.DataFrame(columns=['anio', 'linea', 'mes'])

# Llena el DataFrame con todas las combinaciones de meses y líneas
for mes in meses_del_2025:
    for linea in lineas_transporte:
        nueva_fila = pd.DataFrame([{'anio': anio_2025, 'linea': linea, 'mes': mes}])
        datos_a_predecir = pd.concat([datos_a_predecir, nueva_fila], ignore_index=True)

# --- Preprocesa los nuevos datos ---
X_prediccion = pd.get_dummies(datos_a_predecir, columns=['linea', 'mes'], drop_first=True)

# Asegurarse de que las columnas sean idénticas a las del conjunto de entrenamiento
columnas_entrenamiento = X_preprocesado.columns
X_prediccion = X_prediccion.reindex(columns=columnas_entrenamiento, fill_value=0)

# --- Hace las predicciones SIN SMOTE ---
predicciones_codificadas_sin_smote = modelo_sin_smote.predict(X_prediccion)
predicciones_decodificadas_sin_smote = label_encoder.inverse_transform(predicciones_codificadas_sin_smote)
datos_a_predecir['prediccion_sin_smote'] = predicciones_decodificadas_sin_smote

# --- Hace las predicciones CON SMOTE ---
predicciones_codificadas_con_smote = modelo_final.predict(X_prediccion)
predicciones_decodificadas_con_smote = label_encoder.inverse_transform(predicciones_codificadas_con_smote)
datos_a_predecir['prediccion_con_smote'] = predicciones_decodificadas_con_smote

# --- Muestra los resultados ---
print("Predicciones de afluencia para 2025 (por mes):\n")
print(datos_a_predecir)


# Sin SMOTE la categoría que más aparece es 'Media', tal como comentábamos antes.

# In[12]:


# Obtiene las etiquetas de las clases
clases = label_encoder.classes_

# Genera la matriz de confusión
cm = confusion_matrix(y_test, y_pred_final)

# Crea un DataFrame para una mejor visualización con Seaborn
cm_df = pd.DataFrame(cm, index=clases, columns=clases)

# Crea el gráfico
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges')
plt.title('Matriz de Confusión del Modelo')
plt.ylabel('Valores Reales')
plt.xlabel('Predicciones del Modelo')
# --- Línea clave para guardar la imagen --- colocar antes del show porque sino el gráfico sale en blanco
plt.savefig('matriz_de_confusion_tr.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




