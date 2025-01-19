import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

def train_pred(X, model, year, *, method=0):
    scaler = StandardScaler()
    model[0].fit(scaler.fit_transform(np.array(X)[:-1].T[:-1]), np.array(X)[-1][:-1])
    model[1].fit(scaler.fit_transform(np.array(X)[:-1].T[:-1]), np.array(X)[-1][:-1])
    model[2].fit(scaler.fit_transform(np.array(X)[:-1].T[:-1]), np.array(X)[-1][:-1])
    match method:
        case 0:
            t = (model[0].predict(scaler.fit_transform(np.array(X)[1:].T[:-1])) + model[1].predict(scaler.fit_transform(np.array(X)[1:].T[:-1]))
            + model[2].predict(scaler.fit_transform(np.array(X)[1:].T[:-1]))) / 3
        case 1:
            t = model[0].predict(scaler.fit_transform(np.array(X)[1:].T[:-1]))
        case 2:
            t = model[1].predict(scaler.fit_transform(np.array(X)[1:].T[:-1]))
        case 3:
            t = model[2].predict(scaler.fit_transform(np.array(X)[1:].T[:-1]))
    t = [*t, year]
    X.loc[len(X)] = t
    
    return X

def visual2_0(df, num, *, method=0, column=""):
    fig = plt.figure(figsize=(15,8))
    match method:
        case 0:
            l = [df[i] for i in df.columns]
            year = l[-1]
            l=l[:-1]
            from sklearn.preprocessing import MinMaxScaler
            scal = MinMaxScaler()
            
            for l1 in l:
                l1 = [[i] for i in l1]
                l1 = scal.fit_transform(l1)
                l1 = l1.reshape(1, len(df))
                plt.plot(*l1, marker="o") 

            plt.xticks(ticks=list(range(len(df))), labels=year)
            plt.title(f"{num}")
            plt.ylabel(f"Атрибуты")
            plt.xlabel("Год")
            plt.legend(df.columns[:-1])
        case 1:

            l = [i[column] for i in df]
            year = df[0]["year"]
            for l1 in l:
                l1 = [[i] for i in l1]
                l1 = np.array(l1).reshape(1, len(df[0]))
                plt.plot(*l1, marker="o")

            plt.xticks(ticks=list(range(len(df[0]))), labels=year)
            plt.title(f"{num}")
            plt.ylabel(f"{column}")
            plt.xlabel("Год")
            plt.legend(["RandomForestRegressor", "LinearRegression", "GradientBoostingRegressor"],loc="best")
            plt.grid(True) 
    st.pyplot(fig)

models = [
    RandomForestRegressor(),
    LinearRegression(),
    GradientBoostingRegressor()
]

clusters = {
    'Кластер 0: “Короткие бюджетные поездки”': pd.read_csv('cluster_1.csv'),
    'Кластер 1: “Недорогие поездки средней длины”': pd.read_csv('cluster_2.csv'),
    'Кластер 2: “Длинные премиальные поездки”': pd.read_csv('cluster_3.csv'),
    'Кластер 3: “Дорогие поездки средней длины”': pd.read_csv('cluster_4.csv'),
}


st.title("Прогнозирование для кластеров")

cluster = st.selectbox("Выберите кластер", list(clusters.keys()))

years_ahead = st.number_input("Количество лет для предсказания", min_value=1, max_value=10)

if st.button("Показать график"):
    df = clusters[cluster]
    df_pred = df.copy()

    df_pred = train_pred(df_pred, models, df["year"].iloc[-1] + years_ahead, method=0)

    visual2_0(df_pred, cluster, method=0)
