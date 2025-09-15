#!/usr/bin/env python
# coding: utf-8

# # 📊 Proyecto: Afluencia en el transporte público de la CDMX

# In[48]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# In[2]:


pip install prophet


# In[3]:


def miles(x, pos):
    return f'{int(x/1000)}K'

def filtrar_outliers_percentiles(df, columna, low=0.01, high=0.99):
    p_low = df[columna].quantile(low)
    p_high = df[columna].quantile(high)
    return df[(df[columna] >= p_low) & (df[columna] <= p_high)]


# ## 📂 Cargar Dataset

# In[4]:


f = "C:/Users/xime_/Documents/Proyecto Transportes/Data/afluencia_desglosada_cb_05_2025.csv"
cb = pd.read_csv(f, index_col=0, parse_dates=True,  encoding = 'utf-8').sort_index() #Datos del Cablebus

f = "C:/Users/xime_/Documents/Proyecto Transportes/Data/afluenciamb_desglosado_05_2025.csv"
mb = pd.read_csv(f, index_col=0, parse_dates=True,  encoding = 'utf-8').sort_index() #Datos del metrobus

f = "C:/Users/xime_/Documents/Proyecto Transportes/Data/afluenciastc_desglosado_05_2025.csv"
stc = pd.read_csv(f, index_col=0, parse_dates=True, encoding = 'utf-8').sort_index() #Datos del metro

f = "C:/Users/xime_/Documents/Proyecto Transportes/Data/afluencia_desglosada_trolebus_05_2025.csv"
trolebus = pd.read_csv(f, index_col=0, parse_dates=True,  encoding = 'utf-8').sort_index() #Datos del Trolebus

f = "C:/Users/xime_/Documents/Proyecto Transportes/Data/afluencia_desglosada_tl_05_2025.csv"
tl = pd.read_csv(f, index_col=0, parse_dates=True, encoding = 'utf-8').sort_index() #Datos del Tren Ligero


# ## 🔍 Exploración Inicial

# In[5]:


cb.info()


# In[6]:


mb.info()


# In[7]:


stc.info()


# In[8]:


trolebus.info()


# In[9]:


tl.info()


# ## 🔍📊 EDA

# In[10]:


cb.plot(figsize=(12,10),subplots=True);


# In[11]:


fig, ax = plt.subplots(figsize=(10,5))

# Gráfico de barras por Tipo de Pago
conteo = cb['tipo_pago'].value_counts().sort_index()
ax.bar(conteo.index, conteo.values, color='mediumpurple')
ax.set_title("Tipos de Pago del Cablebus")
ax.set_xlabel("Pago")
ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# In[12]:


cb.describe()


# In[13]:


mb.plot(figsize=(12,10),subplots=True);


# In[14]:


fig, ax = plt.subplots(figsize=(10,5))

# Gráfico de barras por Tipo de Pago
conteo = mb['tipo_pago'].value_counts().sort_index()
ax.bar(conteo.index, conteo.values, color='mediumpurple')
ax.set_title("Tipos de Pago del Metrobús")
ax.set_xlabel("Pago")
ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# In[15]:


mb.describe()


# In[16]:


stc.plot(figsize=(12,10),subplots=True);


# In[17]:


fig, ax = plt.subplots(figsize=(10,5))

# Gráfico de barras por Tipo de Pago
conteo = stc['tipo_pago'].value_counts().sort_index()
ax.bar(conteo.index, conteo.values, color='mediumpurple')
ax.set_title("Tipos de Pago del Metro")
ax.set_xlabel("Pago")
ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# In[18]:


stc.describe()


# In[19]:


trolebus.plot(figsize=(12,10),subplots=True);


# In[20]:


fig, ax = plt.subplots(figsize=(10,5))

# Gráfico de barras por Tipo de Pago
conteo = trolebus['tipo_pago'].value_counts().sort_index()
ax.bar(conteo.index, conteo.values, color='mediumpurple')
ax.set_title("TipoS de Pago del Trolebus")
ax.set_xlabel("Pago")
ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# In[21]:


trolebus.describe()


# In[22]:


tl.plot(figsize=(12,10),subplots=True);


# In[23]:


fig, ax = plt.subplots(figsize=(10,5))

# Gráfico de barras por Tipo de Pago
conteo = tl['tipo_pago'].value_counts().sort_index()
ax.bar(conteo.index, conteo.values, color='mediumpurple')
ax.set_title("Tipos de Pago del Tren Ligero")
ax.set_xlabel("Pago")
ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# In[24]:


tl.describe()


# ## 🧼 Limpieza y Procesamiento de Datos
# 1. Eliminar variables mes, anio y linea.
# 2. Convertir tipo_pago  de object a category.
# 3. Convertir afluencia de float a int.
# 4. Ajustar el rango de los datos del 2022-01-01 al 2025-05-31.
# 5. Analizar valores nulos y determinar su tratamiento.
# 6. Detectar valores atípicos y eliminarlos usando percentiles.
# 7. Usar MinMax para normalizar la variable afluencia.
# 8. Usar One Hot Encoding en la variable tipo_pago.

# ### 🔄 Tipos de datos

# In[25]:


# Convertir tipo_pago de object a category
dataframes = [cb, mb, stc, trolebus, tl]

for df in dataframes:
    df['tipo_pago'] = df['tipo_pago'].astype('category')

# Convertir afluencia que esta como string a numero

for df in dataframes:
    df['afluencia'] = df['afluencia'].astype(str).str.replace(',', '').str.replace(' ', '')
    df['afluencia'] = pd.to_numeric(df['afluencia'], errors='coerce')


# ### 🔧 Tratamiento de valores nulos

# In[26]:


# % de valores nulos
nulos_cb = cb.isnull().mean() * 100 
nulos_mb = mb.isnull().mean() * 100
nulos_stc = stc.isnull().mean() * 100
nulos_trole = trolebus.isnull().mean() * 100
nulos_tl = tl.isnull().mean() * 100

print(f"% de Valores Nulos del Cablebus:\n {nulos_cb}")
print(f"% de Valores Nulos del Metrobús:\n {nulos_mb}")
print(f"% de Valores Nulos del Metro:\n {nulos_stc}")
print(f"% de Valores Nulos del Trolebus:\n {nulos_trole}")
print(f"% de Valores Nulos del Tren Ligero:\n {nulos_tl}")


# ### 🗓️🔎 Seleccionar Rangos de Fecha
# Para el caso de los datasets de mb y stc debemos cortar el periodo 2021, dado que los demás datasets van desde 2022 a 2025. Así tenemos la misma línea del tiempo en todos los datos.

# In[27]:


# Seleccionar para mb y stc un rango de 2022 - 2025
f1 = "2022-01-01"
f2 = "2025-05-31"

mb = mb.loc[f1:f2]
stc = stc.loc[f1:f2]


# ### 🗑️ Eliminar columnas

# In[28]:


# Eliminación de aquellas columnas que no sean tipo_pago y afluencia
columnas = ["tipo_pago", "afluencia"]

cb = cb[columnas]
mb = mb[columnas]
stc = stc[columnas]
trolebus = trolebus[columnas]
tl = tl[columnas]


# ### 🚫 Valores Inconsistentes
# Usaremos un gráfico de boxplot para identificar los valores atípicos de la variable afluencia.

# In[29]:


formatter = FuncFormatter(miles)

fig = plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(6, 5)

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1])
ax3 = fig.add_subplot(gs[:,2])
ax4 = fig.add_subplot(gs[:,3])
ax5 = fig.add_subplot(gs[:,4])

ax1.boxplot(cb.afluencia)
ax2.boxplot(mb.afluencia)
ax3.boxplot(stc.afluencia)
ax4.boxplot(trolebus.afluencia)
ax5.boxplot(tl.afluencia)

ax1.set_title("Cablebus")
ax2.set_title("Metrobús")
ax3.set_title("Metro")
ax4.set_title("Trolebus")
ax5.set_title("Tren Ligero")

# Aplicar formato a miles en cada eje Y
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()


# In[30]:


cb_filtrado = filtrar_outliers_percentiles(cb, 'afluencia')
mb_filtrado = filtrar_outliers_percentiles(mb, 'afluencia')
stc_filtrado = filtrar_outliers_percentiles(stc, 'afluencia')
trolebus_filtrado = filtrar_outliers_percentiles(trolebus, 'afluencia')
tl_filtrado = filtrar_outliers_percentiles(tl, 'afluencia')


# In[31]:


fig = plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(6, 5)

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1])
ax3 = fig.add_subplot(gs[:,2])
ax4 = fig.add_subplot(gs[:,3])
ax5 = fig.add_subplot(gs[:,4])

ax1.boxplot(cb_filtrado.afluencia)
ax2.boxplot(mb_filtrado.afluencia)
ax3.boxplot(stc_filtrado.afluencia)
ax4.boxplot(trolebus_filtrado.afluencia)
ax5.boxplot(tl_filtrado.afluencia)

for ax, title in zip([ax1, ax2, ax3, ax4, ax5], 
                     ["Cablebús", "Metrobús", "Metro", "Trolebús", "Tren Ligero"]):
    ax.set_title(title)
    ax.yaxis.set_major_formatter(formatter)  # Formato a miles

plt.tight_layout()
plt.show()


# ### 🧩 Unir Datos

# In[32]:


T = pd.concat([cb_filtrado, mb_filtrado, stc_filtrado, trolebus_filtrado, tl_filtrado],axis = 0, keys=['cb','mb','metro','trolebus','tl'])
#T = pd.concat([cb, mb, stc, trolebus, tl],axis = 0, keys=['cb','mb','metro','trolebus','tl'])
T.to_csv('C:/Users/xime_/Documents/Proyecto Transportes/CleanData/transportes_limpios.csv')


# ### 📏 Normalizar/Escalar Variables
# 1. Normalizar con el método MinMax la variable afluencia.
# 2. Codificar la variable tipo_pago usando One Hot Encoding.

# In[33]:


# Copia del DataFrame original para no modificarlo
T_fin = T.copy()

# Escalar con MinMax
scaler_minmax = MinMaxScaler()
T_fin[['afluencia']] = scaler_minmax.fit_transform(T_fin[['afluencia']])

# One Hot Encoding
T_fin = pd.get_dummies(T_fin, columns=['tipo_pago'], prefix='pago', drop_first=True)


# In[34]:


ig, ax = plt.subplots(figsize=(12,4))

ax.plot(T_fin.loc['cb']['afluencia'].resample('m').sum(), 'r-', label='cb')
ax.plot(T_fin.loc['mb']['afluencia'].resample('m').sum(), 'k-', label='mb')
ax.plot(T_fin.loc['metro']['afluencia'].resample('m').sum(), 'b-', label='metro')
ax.plot(T_fin.loc['trolebus']['afluencia'].resample('m').sum(), 'g-', label='trolebus')
ax.plot(T_fin.loc['tl']['afluencia'].resample('m').sum(), 'c-', label='tl')

ax.set_title("Afluencia Mensual x Sistema de Transporte")
ax.set_xlabel("Fecha")
ax.set_ylabel("Afluencia")
ax.legend()
plt.tight_layout()


# In[39]:


from statsmodels.tsa.api import VAR
import pandas as pd

# Convertir nivel 'fecha' a datetime y asignar como índice
T_fin.index = pd.to_datetime(T_fin.index.get_level_values('fecha'))

# Agrupar por fecha para eliminar duplicados 
T_fin = T_fin.groupby(T_fin.index).mean()

# Asignar frecuencia diaria y rellenar valores faltantes con forward fill
T_fin = T_fin.asfreq('D').ffill()

# Separar train/test 
train = T_fin.iloc[:-30]
test = T_fin.iloc[-30:]

# Entrenar modelo VAR
model = VAR(train)
selected_lag = model.select_order(15)  # Probar hasta 15 rezagos
best_lag = selected_lag.aic  # o bic/fpe/hqic
print(selected_lag.summary())

# Entrenar con el mejor lag
results = model.fit(selected_lag.aic)

# Predicciones
forecast = results.forecast(train.values, steps=len(test))

# Pasar a DataFrame con mismo formato
forecast_df = pd.DataFrame(forecast, index=test.index, columns=T_fin.columns)

# Evaluar error por transporte
from sklearn.metrics import mean_absolute_error
for col in T_fin.columns:
    mae = mean_absolute_error(test[col], forecast_df[col])
    print(f"{col}: MAE = {mae:.2f}")

# Mostrar primeras predicciones
print(forecast_df.head())


# In[43]:


import pandas as pd
import matplotlib.pyplot as plt

forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)

fechas_test = test.index
fechas_forecast = forecast_df.index

plt.figure(figsize=(12,6))

for col in test.columns:
    plt.plot(fechas_test, test[col], label=f'Real {col}')
    plt.plot(fechas_forecast, forecast_df[col], '--', label=f'Predicción {col}')

plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Predicción vs Real')
plt.legend()
plt.show()



# In[42]:


from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Dividir en train/test 
n_test = 30
train, test = T_fin[:-n_test], T_fin[-n_test:]

# Entrenar modelo VAR con orden p
p = 8  # o 15 según tu criterio
model = VAR(train)
model_fitted = model.fit(p)

# Hacer forecast
forecast_input = train.values[-p:]
forecast = model_fitted.forecast(y=forecast_input, steps=n_test)

# Convertir a DataFrame para comparar
forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)

# Calcular MAE para cada variable
for col in test.columns:
    mae = mean_absolute_error(test[col], forecast_df[col])
    print(f"{col}: MAE = {mae:.5f}")



# In[44]:


T_fin.head(5)


# In[52]:


from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Copia para no modificar original
T_fin = T.copy()

# Escalar 'afluencia'
scaler_minmax = MinMaxScaler()
T_fin[['afluencia']] = scaler_minmax.fit_transform(T_fin[['afluencia']])

# One Hot Encoding en 'tipo_pago'
T_fin = pd.get_dummies(T_fin, columns=['tipo_pago'], prefix='pago', drop_first=True)

# Corregir índice MultiIndex extrayendo el nivel 'fecha'
fechas = pd.to_datetime(T_fin.index.get_level_values('fecha'))
T_fin.index = fechas

# Dividir en train/test
n_test = 30
train, test = T_fin[:-n_test], T_fin[-n_test:]

# Entrenar VAR
p = 8
model = VAR(train)
model_fitted = model.fit(p)

# Forecast
forecast_input = train.values[-p:]
forecast = model_fitted.forecast(y=forecast_input, steps=n_test)
forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)

# Invertir MinMax en 'afluencia'
forecast_df['afluencia'] = scaler_minmax.inverse_transform(forecast_df[['afluencia']])
test_inv = test.copy()
test_inv['afluencia'] = scaler_minmax.inverse_transform(test[['afluencia']])

# Calcular MAE
for col in test.columns:
    mae = mean_absolute_error(test_inv[col], forecast_df[col])
    print(f"{col}: MAE = {mae:.5f}")

# Gráfica de afluencia real vs predicción
plt.figure(figsize=(12,6))
plt.plot(test_inv.index, test_inv['afluencia'], label='Real afluencia')
plt.plot(forecast_df.index, forecast_df['afluencia'], '--', label='Predicción afluencia')
plt.xlabel('Fecha')
plt.ylabel('Afluencia')
plt.legend()
plt.title('Predicción vs Real - Afluencia')
plt.grid(True)
plt.show()



# In[51]:


# Convertir booleanos a int en todo el DataFrame
T_fin['pago_Gratuidad'] = T_fin['pago_Gratuidad'].astype(int)
T_fin['pago_Prepago'] = T_fin['pago_Prepago'].astype(int)

# Ahora sí dividir
n_test = 30
train, test = T_fin[:-n_test], T_fin[-n_test:]

# Entrenar VAR
model = VAR(train)
model_fitted = model.fit(p)


# In[53]:


# Asegurar índice datetime simple
train.index = pd.to_datetime(train.index)

# Convertir booleanos a int 
train['pago_Gratuidad'] = train['pago_Gratuidad'].astype(int)
train['pago_Prepago'] = train['pago_Prepago'].astype(int)

# Convertir todo a float
train = train.astype(float)

# Eliminar filas con NaNs
train = train.dropna()

# Comprobar
print(train.dtypes)
print(train.index)
print(train.head())

# Entrenar VAR
model = VAR(train)
model_fitted = model.fit(p)


# In[54]:


for transporte in ['cb', 'mb', 'metro', 'trolebus', 'tl']:
    df_transp = T.loc[transporte]  # índice solo fecha
    df_transp = df_transp.groupby(df_transp.index).mean()  # agrega duplicados

    # Convertir booleanos a int 
    for col in df_transp.select_dtypes(include='bool').columns:
        df_transp[col] = df_transp[col].astype(int)

    # Convertir todo a float
    df_transp = df_transp.astype(float)

    # Entrenar VAR
    model = VAR(df_transp)
    model_fitted = model.fit(p)

    print(f"Modelo VAR entrenado para {transporte}")


# In[56]:


print(df_transp.dtypes)
print(df_transp.head())

# Selecciona solo columnas numéricas para la agregación
df_numeric = df_transp.select_dtypes(include=['number'])

# Ahora agrupar y promediar solo numéricas
df_grouped = df_numeric.groupby(df_numeric.index).mean()


# In[57]:


for transporte in ['cb', 'mb', 'metro', 'trolebus', 'tl']:
    df_transp = T.loc[transporte]  # Índice solo fecha

    # Convertir booleanos a int
    for col in df_transp.select_dtypes(include='bool').columns:
        df_transp[col] = df_transp[col].astype(int)

    # Seleccionar solo numéricas para agrupar
    df_numeric = df_transp.select_dtypes(include=['number'])

    # Agrupar y promediar
    df_grouped = df_numeric.groupby(df_numeric.index).mean()

    model = VAR(df_grouped)
    model_fitted = model.fit(p)

    print(f"Modelo VAR entrenado para {transporte}")


# In[58]:


df_transp = T.loc['cb']

# Convertir bool a int antes de seleccionar columnas numéricas
for col in df_transp.select_dtypes(include='bool').columns:
    df_transp[col] = df_transp[col].astype(int)

# Seleccionar solo columnas numéricas
df_numeric = df_transp.select_dtypes(include=['number'])

print("Columnas numéricas:", df_numeric.columns)
print("Cantidad columnas:", len(df_numeric.columns))

# Agrupar por fecha y sacar promedio 
df_grouped = df_numeric.groupby(df_numeric.index).mean()

# Confirmar índice con frecuencia diaria
df_grouped = df_grouped.asfreq('D')

print(df_grouped.head())

if len(df_grouped.columns) > 1:
    model = VAR(df_grouped)
    model_fitted = model.fit(p)
else:
    print("No hay suficientes variables para VAR.")


# In[59]:


from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

series = df_grouped['afluencia']

# Dividir en train y test 
n_test = 30
train, test = series[:-n_test], series[-n_test:]

# Definir orden (p,d,q) 
order = (5, 1, 0)

# Entrenar modelo ARIMA
model = ARIMA(train, order=order)
model_fit = model.fit()

# Forecast para test
forecast = model_fit.forecast(steps=n_test)

# Graficar resultados
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test (real)')
plt.plot(test.index, forecast, label='Forecast ARIMA')
plt.legend()
plt.show()

# Imprimir resumen del modelo
print(model_fit.summary())


# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

series = df_grouped['afluencia']

# Partición train / test (últimos 30 días para test)
n_test = 30
train, test = series[:-n_test], series[-n_test:]

# 3. Transformación logarítmica para estabilizar varianza
train_log = np.log1p(train)
test_log = np.log1p(test)

# Parametros SARIMA
auto_model = pm.auto_arima(train_log,
                           start_p=1, start_q=1,
                           max_p=5, max_q=5,
                           start_P=0, max_P=2, max_Q=2,
                           seasonal=True,
                           m=7,                 # patron semanal
                           d=None, D=None,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print(auto_model.summary())

# Ajustar SARIMA con parametros
p, d, q = auto_model.order
P, D, Q, m = auto_model.seasonal_order

model_sarima = SARIMAX(train_log,
                       order=(p, d, q),
                       seasonal_order=(P, D, Q, m),
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarima_fit = model_sarima.fit(disp=False)

# Pronostico a escala log
forecast_log = sarima_fit.get_forecast(steps=len(test_log))
pred_ci_log = forecast_log.conf_int()

# Invertir
forecast = np.expm


# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1. Serie temporal
series = df_grouped['afluencia']

# 2. Partición train / test 
n_test = 30
train, test = series[:-n_test], series[-n_test:]

# 3. Transformación logarítmica para estabilizar varianza
train_log = np.log1p(train)
test_log = np.log1p(test)

# 4. Búsqueda automática de parámetros SARIMA
auto_model = pm.auto_arima(train_log,
                           start_p=1, start_q=1,
                           max_p=5, max_q=5,
                           start_P=0, max_P=2, max_Q=2,
                           seasonal=True,
                           m=7,                 # patrón semanal para datos diarios
                           d=None, D=None,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print(auto_model.summary())

# 5. Ajustar SARIMA con parámetros óptimos
p, d, q = auto_model.order
P, D, Q, m = auto_model.seasonal_order

model_sarima = SARIMAX(train_log,
                       order=(p, d, q),
                       seasonal_order=(P, D, Q, m),
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarima_fit = model_sarima.fit(disp=False)

# 6. Pronóstico en escala log
forecast_log = sarima_fit.get_forecast(steps=len(test_log))
pred_ci_log = forecast_log.conf_int()

# 7. Invertir transformación logarítmica para obtener pronóstico en escala original
forecast = np.expm1(forecast_log.predicted_mean)
lower_ci = np.expm1(pred_ci_log.iloc[:, 0])
upper_ci = np.expm1(pred_ci_log.iloc[:, 1])

# --- Visualizaciones ---

plt.figure(figsize=(14,6))

# Serie original completa
plt.plot(series.index, series, label='Serie original')

# Ajuste dentro de la muestra (train)
fitted_values = np.expm1(sarima_fit.fittedvalues)
plt.plot(train.index, fitted_values, color='red', label='Ajuste SARIMA (train)')

# Pronóstico en test
plt.plot(test.index, forecast, color='green', label='Pronóstico SARIMA (test)')

# Intervalo de confianza
plt.fill_between(test.index, lower_ci, upper_ci, color='lightgreen', alpha=0.3)

plt.title('Modelo SARIMA: Serie original, ajuste y pronóstico')
plt.legend()
plt.show()

# Visualización residuos
residuals = sarima_fit.resid

plt.figure(figsize=(12,4))
plt.plot(residuals)
plt.title('Residuos del modelo SARIMA')
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals.dropna(), lags=40)
plt.title('ACF de residuos SARIMA')
plt.show()


# In[77]:


import matplotlib.dates as mdates

plt.figure(figsize=(14,6))

# Serie original completa
plt.plot(series.index, series, label='Serie original')

# Ajuste dentro de la muestra (train)
fitted_values = np.expm1(sarima_fit.fittedvalues)
plt.plot(train.index, fitted_values, color='red', label='Ajuste SARIMA (train)')

# Pronóstico en test
plt.plot(test.index, forecast, color='green', label='Pronóstico SARIMA (test)')

# Intervalo de confianza
plt.fill_between(test.index, lower_ci, upper_ci, color='lightgreen', alpha=0.3)

# Ajustar límites Y: mínimo y máximo considerando todo el rango que mostramos
y_min = min(series.min(), fitted_values.min(), lower_ci.min())
y_max = max(series.max(), fitted_values.max(), upper_ci.max())
plt.ylim(y_min*0.95, y_max*1.05)  # un pequeño margen arriba y abajo

# Formato de fechas en eje X para que se vean mejor 
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotat)



# In[78]:


import matplotlib.dates as mdates

plt.figure(figsize=(14,6))

# Serie original completa
plt.plot(series.index, series, label='Serie original')

# Ajuste dentro de la muestra (train)
fitted_values = np.expm1(sarima_fit.fittedvalues)
plt.plot(train.index, fitted_values, color='red', label='Ajuste SARIMA (train)')

# Pronóstico en test
plt.plot(test.index, forecast, color='green', label='Pronóstico SARIMA (test)')

# Intervalo de confianza
plt.fill_between(test.index, lower_ci, upper_ci, color='lightgreen', alpha=0.3)

# Ajustar límites Y: mínimo y máximo considerando todo el rango que mostramos
y_min = min(series.min(), fitted_values.min(), lower_ci.min())
y_max = max(series.max(), fitted_values.max(), upper_ci.max())
plt.ylim(y_min*0.95, y_max*1.05)  # un pequeño margen arriba y abajo

# Formato de fechas en eje X para que se vean mejor (usando formato semanal)
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.xlabel('Fecha')
plt.ylabel('Afluencia')
plt.title('Modelo SARIMA: Serie original, ajuste y pronóstico')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[80]:


import matplotlib.dates as mdates
from datetime import timedelta

plt.figure(figsize=(14,6))

# Serie original completa
plt.plot(series.index, series, label='Serie original')

# Ajuste dentro de la muestra (train)
fitted_values = np.expm1(sarima_fit.fittedvalues)
plt.plot(train.index, fitted_values, color='red', label='Ajuste SARIMA (train)')

# Pronóstico en test
plt.plot(test.index, forecast, color='green', label='Pronóstico SARIMA (test)')

# Intervalo de confianza
plt.fill_between(test.index, lower_ci, upper_ci, color='lightgreen', alpha=0.3)

# Ajustar límites Y (con pequeño margen)
y_min = min(series.min(), fitted_values.min(), lower_ci.min())
y_max = max(series.max(), fitted_values.max(), upper_ci.max())
plt.ylim(y_min*0.95, y_max*1.05)

# Limitar eje X a últimos 2 años desde la última fecha de la serie
fecha_final = series.index.max()
fecha_inicio = fecha_final - pd.DateOffset(years=2)
plt.xlim(fecha_inicio, fecha_final + timedelta(days=10))  # un poco de margen a la derecha

# Formato y ticks del eje X: mostrar cada mes (o cada 2 meses si está muy lleno)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.xlabel('Fecha')
plt.ylabel('Afluencia')
plt.title('Modelo SARIMA: Serie original, ajuste y pronóstico (últimos 2 años)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[81]:


import matplotlib.dates as mdates
from datetime import timedelta

plt.figure(figsize=(14,6))

plt.plot(series.index, series, label='Serie original')
plt.plot(train.index, fitted_values, color='red', label='Ajuste SARIMA (train)')
plt.plot(test.index, forecast, color='green', label='Pronóstico SARIMA (test)')
plt.fill_between(test.index, lower_ci, upper_ci, color='lightgreen', alpha=0.3)

# Ajuste eje Y con margen y sin negativos
y_min = min(series.min(), fitted_values.min(), lower_ci.min())
y_max = max(series.max(), fitted_values.max(), upper_ci.max())
y_lower_lim = max(0, y_min * 0.95)
y_upper_lim = y_max * 1.05
plt.ylim(y_lower_lim, y_upper_lim)

fecha_final = series.index.max()
fecha_inicio = fecha_final - pd.DateOffset(years=2)
plt.xlim(fecha_inicio, fecha_final + timedelta(days=10))

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.xlabel('Fecha')
plt.ylabel('Afluencia')
plt.title('Modelo SARIMA: Serie original, ajuste y pronóstico (últimos 2 años)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[84]:


import matplotlib.dates as mdates
from datetime import timedelta

plt.figure(figsize=(14,6))

plt.plot(series.index, series, label='Serie original')
plt.plot(train.index, fitted_values, color='red', label='Ajuste SARIMA (train)')
plt.plot(test.index, forecast, color='green', label='Pronóstico SARIMA (test)')
plt.fill_between(test.index, lower_ci, upper_ci, color='lightgreen', alpha=0.3)

# Ajuste eje Y fijo de 0 a 0.5
plt.ylim(0, 0.00001)

fecha_final = series.index.max()
fecha_inicio = fecha_final - pd.DateOffset(years=2)
plt.xlim(fecha_inicio, fecha_final + timedelta(days=10))

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.xlabel('Fecha')
plt.ylabel('Afluencia')
plt.title('Modelo SARIMA: Serie original, ajuste y pronóstico (últimos 2 años)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[85]:


# Normalizar a [0, 0.5]
min_val = series.min()
max_val = series.max()

series_norm = (series - min_val) / (max_val - min_val) * 0.5
fitted_values_norm = (fitted_values - min_val) / (max_val - min_val) * 0.5
forecast_norm = (forecast - min_val) / (max_val - min_val) * 0.5
lower_ci_norm = (lower_ci - min_val) / (max_val - min_val) * 0.5
upper_ci_norm = (upper_ci - min_val) / (max_val - min_val) * 0.5

plt.figure(figsize=(14,6))
plt.plot(series.index, series_norm, label='Serie normalizada')
plt.plot(train.index, fitted_values_norm, color='red', label='Ajuste SARIMA (train)')
plt.plot(test.index, forecast_norm, color='green', label='Pronóstico SARIMA (test)')
plt.fill_between(test.index, lower_ci_norm, upper_ci_norm, color='lightgreen', alpha=0.3)
plt.ylim(0, 0.5)


# In[86]:


# Normalizar a [0, 0.5]
min_val = series.min()
max_val = series.max()

series_norm = (series - min_val) / (max_val - min_val) * 0.5
fitted_values_norm = (fitted_values - min_val) / (max_val - min_val) * 0.5
forecast_norm = (forecast - min_val) / (max_val - min_val) * 0.5
lower_ci_norm = (lower_ci - min_val) / (max_val - min_val) * 0.5
upper_ci_norm = (upper_ci - min_val) / (max_val - min_val) * 0.5

import matplotlib.dates as mdates
from datetime import timedelta

plt.figure(figsize=(14,6))

plt.plot(series.index, series_norm, label='Serie Original Normalizada [0, 0.5]', color='blue')
plt.plot(train.index, fitted_values_norm, label='Ajuste SARIMA (Train) Normalizado', color='red')
plt.plot(test.index, forecast_norm, label='Pronóstico SARIMA (Test) Normalizado', color='green')
plt.fill_between(test.index, lower_ci_norm, upper_ci_norm, color='lightgreen', alpha=0.3, label='Intervalo Confianza Pronóstico')

# Fijar eje Y entre 0 y 0.5
plt.ylim(0, 0.5)

fecha_final = series.index.max()
fecha_inicio = fecha_final - pd.DateOffset(years=2)
plt.xlim(fecha_inicio, fecha_final + timedelta(days=10))

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.xlabel('Fecha')
plt.ylabel('Afluencia (normalizada)')
plt.title('Modelo SARIMA: Serie, ajuste y pronóstico normalizados (últimos 2 años)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




