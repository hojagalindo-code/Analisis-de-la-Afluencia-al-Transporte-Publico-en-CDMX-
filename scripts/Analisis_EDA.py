#!/usr/bin/env python
# coding: utf-8

# In[175]:


# 1) Importar librerías
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown


# In[127]:


sns.set(style='whitegrid')


# In[128]:


# Función auxiliar para mostrar explicaciones
def md(text):
    display(Markdown(text))


# In[134]:


# 2) Cargar datasets
# --------------------------

files_info = {
    "Cablebús": "/content/afluencia_desglosada_cb_05_2025.csv",
    "Metrobús": "/content/afluenciamb_desglosado_05_2025.csv",
    "Metro": "/content/afluenciastc_desglosado_05_2025.csv",
    "Trolebús": "/content/afluencia_desglosada_trolebus_05_2025.csv",
    "Tren Ligero": "/content/afluencia_desglosada_tl_05_2025.csv"
}


# In[135]:


dataframes = []
for sistema, path in files_info.items():
    df_temp = pd.read_csv(path, index_col=0, parse_dates=True, encoding='utf-8').sort_index()
    df_temp["Sistema"] = sistema
    dataframes.append(df_temp)

# Unir todos los sistemas en un único DataFrame
df = pd.concat(dataframes, ignore_index=False)

md(f"Se han cargado **{len(files_info)} datasets** con un total de **{len(df):,} registros**.")


# In[136]:


md("Revisión general del contenido del dataset")
display(df.head())


# In[137]:


# 3) Información general
# --------------------------
md("## 1) Información general de las variables")
display(df.info())
display(df.describe(include='all'))


# In[138]:


# 2) Normalización de columnas y tipos
# ------------------------
md('## 2) Normalización de nombres y tipos')
# Hacer nombres de columnas consistentes (sin mayúsculas, sin espacios)
orig_cols = df.columns.tolist()
new_cols = []
for c in orig_cols:
    c2 = c.strip().lower()
    c2 = c2.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')
    c2 = c2.replace('ñ','n')
    c2 = c2.replace(' ','_').replace('-','_')
    new_cols.append(c2)

rename_map = dict(zip(orig_cols, new_cols))
df = df.rename(columns=rename_map)
md(f"Se renombraron columnas: `{rename_map}` (muestra).")


# In[139]:


# 2.2) Convertir tipos
if 'anio' in df.columns:
    df['anio'] = pd.to_numeric(df['anio'], errors='coerce').astype('Int64')
if 'afluencia' in df.columns:
    df['afluencia'] = pd.to_numeric(df['afluencia'], errors='coerce')


# In[140]:


# 3) Valores faltantes y duplicados
# ------------------------
md('## 3) Valores faltantes y duplicados')
missing = df.isna().sum().sort_values(ascending=False)
display(missing[missing>0])

dup = df.duplicated().sum()
md(f'Número de filas duplicadas: **{dup}**')

# Mostrar algunas filas con datos faltantes en columnas clave
key_cols = [c for c in ['anio','mes','linea','tipo_pago','afluencia'] if c in df.columns]
if len(key_cols)>0:
    display(df[df[key_cols].isna().any(axis=1)].head(8))


# In[141]:


df = df.drop(columns=['estacion', 'linea'], errors='ignore')
display(df.isna().sum())


# In[147]:


md("## 4) Seleccionar rangos de fecha")
# Filtrar solo datos desde 2022 en adelante
df = df[df["anio"] >= 2022]


# In[150]:


# 4) Resumen estadístico
# ------------------------
md('## 5) Resumen estadístico')
if 'afluencia' in df.columns:
    display(df['afluencia'].describe().to_frame().T)
    md('Distribución (cuartiles) de Afluencia mostrada arriba.')


# In[151]:


# Categóricas: conteo
cat_cols = [c for c in ['linea','tipo_pago'] if c in df.columns]
for c in cat_cols:
    md(f'### Conteo y valores únicos para `{c}`')
    display(df[c].value_counts(dropna=False).head(15))
    md(f'Número de valores únicos en `{c}`: {df[c].nunique(dropna=True)}')


# In[152]:


# 5) Visualizaciones
# ------------------------
md('## 5) Visualizaciones\nA continuación se generan gráficos clave: histogramas, boxplots, barras y series temporales.')

# 5.1 Histograma de afluencia
if 'afluencia' in df.columns:
    plt.figure(figsize=(10,4))
    sns.histplot(df['afluencia'].dropna(), kde=False)
    plt.title('Histograma: Afluencia (sin transformación)')
    plt.xlabel('Afluencia')
    plt.show()

    plt.figure(figsize=(10,4))
    sns.histplot(np.log1p(df['afluencia'].dropna()), kde=False)
    plt.title('Histograma: log(1 + Afluencia) — para ver asimetría')
    plt.xlabel('log1p(afluencia)')
    plt.show()


# In[153]:


print(df['afluencia'].describe())
print(df['afluencia'].isna().sum(), "valores nulos")


# In[154]:


# Graficar solo valores positivos
df_positivos = df[df['afluencia'] > 0]

# Combinar filtro de outliers + eliminación de ceros
q99 = df_positivos['afluencia'].quantile(0.99)
df_filtrado = df_positivos[df_positivos['afluencia'] <= q99]


# In[155]:


plt.figure(figsize=(12,5))
sns.histplot(df_filtrado['afluencia'], bins=50, kde=False)
plt.title("Histograma: Afluencia (>0 y sin valores extremos)")
plt.xlabel("Afluencia")
plt.ylabel("Frecuencia")
plt.show()

plt.figure(figsize=(12,5))
sns.histplot(np.log1p(df_filtrado['afluencia']), bins=50, kde=False)
plt.title("Histograma: log(1 + Afluencia) — sin ceros y sin valores extremos")
plt.xlabel("log(1 + afluencia)")
plt.ylabel("Frecuencia")
plt.show()


# In[157]:


# 6) Análisis temporal de afluencia (Afluencia mensual total por sistema)
# --------------------------
afluencia_mensual = df.groupby(['sistema', 'anio', 'mes'])['afluencia'].sum().reset_index()

import matplotlib.pyplot as plt

for sistema in afluencia_mensual['sistema'].unique():
    df_sistema = afluencia_mensual[afluencia_mensual['sistema'] == sistema]
    plt.figure()
    plt.plot(df_sistema['mes'].astype(str) + '-' + df_sistema['anio'].astype(str), df_sistema['afluencia'])
    plt.xticks(rotation=45)
    plt.title(f"Afluencia mensual - {sistema}")
    plt.ylabel("Número de usuarios")
    plt.xlabel("Mes-Año")
    plt.tight_layout()
    plt.show()


# In[169]:


# 5.2 Comparación entre tipos de pago
tipo_pago = df.groupby(['sistema', 'tipo_pago'])['afluencia'].sum().reset_index()

import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(data=tipo_pago, x='sistema', y='afluencia', hue='tipo_pago')
plt.title("Afluencia total por tipo de pago y sistema")
plt.ylabel("Total de usuarios")
plt.xlabel("Sistema de transporte")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[161]:


# 5.5 Scatter: Afluencia vs Mes (muestras)
if 'mes' in df.columns and 'afluencia' in df.columns:
    plt.figure(figsize=(10,4))
    sns.scatterplot(data=df.sample(frac=min(1, 5000/len(df))) if len(df)>5000 else df, x='mes', y='afluencia', alpha=0.6)
    plt.title('Dispersion: mes vs afluencia (muestra si dataset grande)')
    plt.show()


# In[162]:


# 6) Detección de atípicos
# ------------------------
md('## 6) Detección de atípicos (IQR)')
if 'afluencia' in df.columns:
    q1 = df['afluencia'].quantile(0.25)
    q3 = df['afluencia'].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    md(f'IQR: Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f} — límites: [{low:.1f}, {high:.1f}]')
    outliers = df[(df['afluencia'] < low) | (df['afluencia'] > high)].copy()
    md(f'Número de filas detectadas como atípicas por IQR: **{len(outliers)}**')
    display(outliers.sort_values('afluencia', ascending=False).head(20))

    # Marcar en el dataframe si conviene
    df['is_outlier_iqr'] = False
    df.loc[outliers.index, 'is_outlier_iqr'] = True


# In[163]:


# 7) Análisis agrupado y comparativas
# ------------------------
md('## 7) Análisis agrupado: por Línea y por Tipo de Pago')
if 'linea' in df.columns and 'afluencia' in df.columns:
    agg_linea = df.groupby('linea')['afluencia'].agg(['sum','mean','median','count']).sort_values('sum', ascending=False)
    display(agg_linea.head(20))

if 'tipo_pago' in df.columns and 'afluencia' in df.columns:
    agg_pago = df.groupby('tipo_pago')['afluencia'].agg(['sum','mean','median','count']).sort_values('sum', ascending=False)
    display(agg_pago)

# Comparativa línea vs tipo de pago (pivot)
if set(['linea','tipo_pago','afluencia']).issubset(df.columns):
    md('Pivot: Afluencia media por Línea (filas) y Tipo de Pago (columnas) — mostrar top 10 líneas por total')
    top10 = agg_linea.head(10).index.tolist()
    pivot_lp = df[df['linea'].isin(top10)].pivot_table(index='linea', columns='tipo_pago', values='afluencia', aggfunc='mean')
    display(pivot_lp)


# In[164]:


# 8) Correlaciones (numéricas)
# ------------------------
md('## 8) Correlaciones entre variables numéricas')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 2:
    display(df[num_cols].corr())
    plt.figure(figsize=(6,5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f')
    plt.title('Matriz de correlación (numéricas)')
    plt.show()
else:
    md('No hay suficientes variables numéricas para correlación más allá de afluencia.')


# In[171]:


md('## 9) Análisis de tendencias mensuales')

afluencia_mensual = df.groupby(['sistema', 'anio', 'mes'])['afluencia'].sum().reset_index()
afluencia_mensual = afluencia_mensual.sort_values(['sistema', 'anio', 'mes'])

sistemas = afluencia_mensual['sistema'].unique()

for sistema in sistemas:
    df_sis = afluencia_mensual[afluencia_mensual['sistema'] == sistema]

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_sis, x='mes', y='afluencia', hue='anio', marker='o')
    plt.title(f"Tendencia mensual de afluencia - {sistema}")
    plt.xlabel("Mes")
    plt.ylabel("Afluencia total")
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title='Año')
    plt.show()


# In[172]:


# 10) Guardar resultados intermedios
# ------------------------
md('## 10) Guardar resultados intermedios (opcional)')
# Guardar tablas agregadas a CSV en el directorio actual
agg_out_files = []
if 'agg_linea' in locals():
    agg_linea.reset_index().to_csv('agg_linea.csv', index=False)
    agg_out_files.append('agg_linea.csv')
if 'agg_pago' in locals():
    agg_pago.reset_index().to_csv('agg_pago.csv', index=False)
    agg_out_files.append('agg_pago.csv')
if 'pivot_lp' in locals():
    pivot_lp.reset_index().to_csv('pivot_linea_tipopago_top10.csv', index=False)
    agg_out_files.append('pivot_linea_tipopago_top10.csv')

if agg_out_files:
    md('Se han guardado los siguientes archivos en el directorio de trabajo:')
    display(agg_out_files)
else:
    md('No se generaron archivos agregados para guardar.')


# In[173]:


# 10) Conclusiones rápidas (automáticas) — ejemplos
# ------------------------
md('## 10) Conclusiones rápidas — sugerencias automáticas')
insights = []
if 'afluencia' in df.columns:
    mean_af = df['afluencia'].mean()
    median_af = df['afluencia'].median()
    insights.append(f'La afluencia media es {mean_af:.1f} y la mediana {median_af:.1f} (si media >> mediana hay asimetría).')

if 'periodo' in df.columns:
    # Mes con mayor afluencia total
    mes_top = df.groupby('mes')['afluencia'].sum().idxmax()
    insights.append(f'El mes (numérico) con mayor afluencia total registrada es: {mes_top}.')

if 'linea' in df.columns and 'afluencia' in df.columns:
    top_line = df.groupby('linea')['afluencia'].sum().idxmax()
    insights.append(f'La línea con mayor afluencia total es: {top_line}.')

for s in insights:
    md('- ' + s)

