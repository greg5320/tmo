import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title='Прогноз оценки вина', layout='wide')

@st.cache_data
def load_data():
    df = pd.read_csv('wine_reviews.csv')
    df = df.dropna(subset=['points', 'price'])
    for col in ['designation','region_1','region_2']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    return df

@st.cache_resource
def build_preprocessor(df):
    features = ['country','province','region_1','winery','price']
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    cat_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, ['price']),
        ('cat', cat_transformer, ['country','province','region_1','winery'])
    ])
    preprocessor.fit(df[features])
    return preprocessor, features

def train_model(df, preprocessor, n_estimators, max_depth, min_samples_split):
    features = ['country','province','region_1','winery','price']
    X = df[features]
    y = df['points']
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return pipeline, (r2, mae, rmse)

df = load_data()
preprocessor, features = build_preprocessor(df)

st.title('🍷 Прогноз оценки вина')
st.write('Интерактивное приложение для предсказания баллов вина на основе характеристик.')

st.sidebar.header("Гиперпараметры RandomForestRegressor")
n_estimators = st.sidebar.slider(
    'Количество деревьев (n_estimators)',
    min_value=10, max_value=300, value=100, step=10
)
max_depth = st.sidebar.select_slider(
    'Максимальная глубина (max_depth)',
    options=[5, 10, 15, 20, 25, 30, None],
    value=None
)
min_samples_split = st.sidebar.slider(
    'Минимум сэмплов для сплита (min_samples_split)',
    min_value=2, max_value=20, value=2, step=1
)

if st.sidebar.button('Обучить модель и оценить'):
    with st.spinner('Обучение модели...'):
        model_pipeline, (r2, mae, rmse) = train_model(
            df, preprocessor,
            n_estimators, max_depth, min_samples_split
        )
    st.success('Модель обучена! Метрики на тестовой выборке:')
    col1, col2, col3 = st.columns(3)
    col1.metric("R² (Коэфф. детерминации)", f"{r2:.4f}")
    col2.metric("MAE (Сред. абс. ошибка)", f"{mae:.2f} балла")
    col3.metric("RMSE (Корень из ср.кв. ошибки)", f"{rmse:.2f} балла")
else:
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100, max_depth=None,
            min_samples_split=2, random_state=42, n_jobs=-1
        ))
    ])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    model_pipeline.fit(df_train[features], df_train['points'])
    st.info("Настройте гиперпараметры и нажмите «Обучить модель и оценить».")

st.subheader('Введите параметры вина для прогноза оценки')
price = st.slider('Цена (USD)', int(df['price'].min()), int(df['price'].max()), 50)
country = st.selectbox('Страна', sorted(df['country'].dropna().unique()))
province = st.selectbox('Провинция', sorted(df[df['country']==country]['province'].dropna().unique()))
region = st.selectbox('Регион', sorted(df[df['province']==province]['region_1'].dropna().unique()))
winery = st.selectbox('Винодельня', sorted(df[df['region_1']==region]['winery'].dropna().unique()))

input_df = pd.DataFrame([{
    'price': price,
    'country': country,
    'province': province,
    'region_1': region,
    'winery': winery
}], columns=features)

if st.button('Прогноз'):
    pred = model_pipeline.predict(input_df)[0]
    st.success(f"Прогнозируемая оценка вина: {pred:.2f} баллов")

