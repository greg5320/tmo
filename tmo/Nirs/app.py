import streamlit as st
st.set_page_config(page_title='Прогноз оценки вина', layout='wide')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


@st.cache_data
def load_and_train():
    df = pd.read_csv('wine_reviews.csv')
    df = df.dropna(subset=['points','price'])
    for col in ['designation','region_1','region_2']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    features = ['country','province','region_1','winery','price']
    X = df[features]
    y = df['points']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, ['price']),
        ('cat', cat_transformer, ['country','province','region_1','winery'])
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return df, features, model


df, features, model = load_and_train()

st.title('🍷 Прогноз оценки вина')
st.write('Интерактивное приложение для предсказания баллов вина на основе характеристик.')

price = st.slider('Цена (USD)', int(df['price'].min()), int(df['price'].max()), 50)
country = st.selectbox('Страна', sorted(df['country'].dropna().unique()))
province = st.selectbox('Провинция', sorted(df[df['country']==country]['province'].dropna().unique()))
region = st.selectbox('Регион', sorted(df[df['province']==province]['region_1'].dropna().unique()))
winery = st.selectbox('Винодельня', sorted(df[df['region_1']==region]['winery'].dropna().unique()))

input_df = pd.DataFrame({
    'price': [price],
    'country': [country],
    'province': [province],
    'region_1': [region],
    'winery': [winery]
})

if st.button('Прогноз'):  
    pred = model.predict(input_df)[0]
    st.success(f"Прогнозируемая оценка вина: {pred:.2f} баллов")