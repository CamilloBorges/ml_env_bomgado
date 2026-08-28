from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.hooks.base import BaseHook
import pandas as pd
import numpy as np
import re
import os
import joblib

# Bibliotecas de ML e Preprocessamento
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import flaml
from flaml import AutoML
import mlflow

# Biblioteca para ler e gravar Delta nativamente sem Spark
from deltalake import DeltaTable, write_deltalake

# ==========================================
# CONFIGURAÇÕES E PARÂMETROS
# ==========================================
# Ajuste os caminhos abaixo para o seu Storage Account / OneLake
LAKEHOUSE_URI_BRONZE = "abfss://Bomgado@onelake.dfs.fabric.microsoft.com/armazem_lh.Lakehouse/Tables/dbo/bronze_treinamento"
LAKEHOUSE_URI_PREVISAO = "abfss://Bomgado@onelake.dfs.fabric.microsoft.com/armazem_lh.Lakehouse/Tables/dbo/previsao_demanda_diaria_v2"

# Configurações do FLAML (Perfil Padrão)
FLAML_SETTINGS = {
    "time_budget": 180, # 5 minutos
    "estimator_list": ["lgbm", "xgboost"],
    "task": "ts_forecast",
    "log_file_name": "/opt/airflow/logs/flaml_experiment.log",
    "log_type": "better",
    "max_iter": 10,
    "n_jobs": 1,
    "seed": 41,
    "verbose": 1,
    "featurization": "auto",
}

# ==========================================
# FUNÇÕES DE PREPROCESSAMENTO (Do script original)
# ==========================================
def _coerce_na_for_sklearn(df):
    df = df.copy()
    for col in df.columns:
        s = df[col]
        dt = s.dtype
        is_object_like = (isinstance(dt, pd.CategoricalDtype) or dt == object or dt == 'boolean' or str(dt) == 'string')
        if is_object_like:
            s_obj = s.astype(object)
            df[col] = s_obj.where(s_obj.notna(), np.nan)
    return df

def create_fillna_processor(df):
    mean_features, median_features, mode_features = [], [], []
    
    mean_features = [col for col in df.select_dtypes(include=["number"], exclude=["timedelta"]).columns if df[col].skew(skipna=True) <= 1]
    median_features = [col for col in df.select_dtypes(include=["number"], exclude=["timedelta"]).columns if df[col].skew(skipna=True) > 1]
    
    datetime_features = df.select_dtypes(include=["datetime", "timedelta"]).columns.tolist()
    all_features = mean_features + median_features
    mode_features = [col for col in df.columns.tolist() if col not in all_features + datetime_features]

    transformers = []
    if mean_features:
        transformers.append(("mean_imputer", SimpleImputer(strategy="mean"), mean_features))
    if median_features:
        transformers.append(("median_imputer", SimpleImputer(strategy="median"), median_features))
    if mode_features:
        transformers.append(("mode_imputer", SimpleImputer(strategy="most_frequent"), mode_features))

    column_transformer = ColumnTransformer(transformers=transformers)
    all_features = mean_features + median_features + mode_features

    return column_transformer.fit(_coerce_na_for_sklearn(df)), all_features, datetime_features

def fillna(df, processor, all_features, datetime_features):
    filled_array = processor.transform(_coerce_na_for_sklearn(df))
    filled_df = pd.DataFrame(filled_array, columns=all_features, index=df.index)
    if datetime_features:
        datetime_data = df[datetime_features].ffill()
        filled_df = pd.concat([datetime_data, filled_df], axis=1)
    
    mode_vals = filled_df.mode()
    if not mode_vals.empty:
        filled_df = filled_df.fillna(mode_vals.iloc[0])
    return filled_df

def resample_item(group: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
    group = group.set_index(time_col).sort_index()
    full_index = pd.date_range(group.index.min(), group.index.max(), freq="D")
    group = group.reindex(full_index)
    
    y = group[target_col].fillna(0)
    X_feats = group.drop(columns=[target_col]).ffill().bfill()
    
    out = X_feats.copy()
    out[target_col] = y
    out = out.reset_index().rename(columns={"index": time_col})
    return out

# ==========================================
# PIPELINE PRINCIPAL DO AIRFLOW
# ==========================================
def executar_pipeline_demanda():
    print("[Etapa 0] Buscando credenciais no painel do Airflow...")
    # Puxa a conexão "Datalake" exatamente como você nomeou no print
    conexao = BaseHook.get_connection("Datalake")
    
    STORAGE_OPTIONS = {
        "tenant_id": conexao.extra_dejson.get("tenantId"),
        "client_id": conexao.login,     # No Airflow, o 'Client ID' é salvo no campo login
        "client_secret": conexao.password # O 'Secret' é salvo no campo password
    }
    print("[Etapa 1] Conectando ao MLflow via disco local...")
    mlflow.set_tracking_uri("sqlite:////mlruns/mlflow.db")
    mlflow.set_experiment("IA_Previsao_Demanda_Acougue")

    print("[Etapa 2] Carregando dados históricos via deltalake...")
    dt = DeltaTable(LAKEHOUSE_URI_BRONZE, storage_options=STORAGE_OPTIONS)
    df = dt.to_pandas()
    
    # Padronizar nomes e manter apenas histórico
    df = df.rename(columns=lambda c: re.sub('[^A-Za-z0-9_]+', '_', c))
    df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce')
    df['ds'] = pd.to_datetime(df['ds'])
    df = df[df['ds'] >= '2024-01-01'] # Exemplo: Mantém dados a partir de 2024
    top_skus = df['eanBalanca'].value_counts().head(50).index # Ajuste o 'head' se quiser mais ou menos produtos
    df = df[df['eanBalanca'].isin(top_skus)].copy()

    df_hist = df[df['quantidade'].notnull()].copy()
    print(f"Linhas carregadas para treino: {len(df_hist)}")

    time_col = "ds"
    item_col = "eanBalanca"
    target_col = "quantidade"

    df_hist[time_col] = pd.to_datetime(df_hist[time_col])
    if item_col in df_hist.columns:
        df_hist[item_col] = df_hist[item_col].astype("category")

    # Split e Preparação
    X = df_hist.select_dtypes(include=["number", "datetime", "category", "boolean"], exclude=["timedelta"])
    
    X_sorted = X.sort_values(time_col)
    unique_dates = np.sort(X_sorted[time_col].unique())
    cutoff_date = unique_dates[int(len(unique_dates) * 0.8)]

    X_train = X_sorted[X_sorted[time_col] <= cutoff_date].copy()
    y_train = X_train.pop(target_col)
    
    # Treinar Preprocessor
    preprocessor, all_features, datetime_features = create_fillna_processor(X_train)
    X_train_filled = fillna(X_train, preprocessor, all_features, datetime_features)
    
    # Resample diário por item
    Xy_train = X_train_filled.copy()
    Xy_train[target_col] = y_train
    
    print("[Etapa 3] Resample diário por SKU...")
    Xy_resampled = Xy_train.groupby(item_col, group_keys=False).apply(
        lambda g: resample_item(g, time_col, target_col)
    ).reset_index(drop=True)

    print("[Etapa 4] Iniciando AutoML (FLAML)...")
    automl = AutoML()
    with mlflow.start_run() as run:
        automl.fit(
            dataframe=Xy_resampled,
            label=target_col,
            period=25,
            time_col=time_col,
            group_id=item_col,
            **FLAML_SETTINGS
        )
        
        # Registrar o modelo no MLflow
        mlflow.sklearn.log_model(automl, "model", registered_model_name="IA_Previsao_Demanda_Acougue")
        print(f"Modelo registrado com sucesso. Run ID: {run.info.run_id}")

    # ==========================================
    # INFERÊNCIA FUTURA
    # ==========================================
    print("[Etapa 5] Gerando previsões futuras...")
    df_fut = df[df[time_col] > pd.Timestamp.now()].copy()
    
    if df_fut.empty:
        print("Nenhuma data futura encontrada em bronze_treinamento. Encerrando.")
        return

    df_fut[time_col] = pd.to_datetime(df_fut[time_col])
    if item_col in df_fut.columns:
        df_fut[item_col] = df_fut[item_col].astype("category")

    expected_cols = list(all_features) + list(datetime_features)
    for col_name in expected_cols:
        if col_name not in df_fut.columns:
            df_fut[col_name] = np.nan

    X_fut_for_imputer = df_fut[expected_cols]
    X_fut_full = fillna(X_fut_for_imputer, preprocessor, all_features, datetime_features)
    
    feature_cols = list(X_train.columns)
    X_fut_features = X_fut_full[feature_cols]

    # Previsão em massa (sem necessidade de lotes, o Pandas faz tudo na RAM tranquilamente)
    previsoes = automl.predict(X_fut_features)

    df_pred_final = pd.DataFrame({
        "ds": X_fut_full[time_col].values,
        "eanBalanca": X_fut_full[item_col].astype(str).values,
        "quantidade_prevista": previsoes,
    })

    print("[Etapa 6] Gravando previsões de volta no Lakehouse (Delta)...")
    write_deltalake(
        LAKEHOUSE_URI_PREVISAO,
        df_pred_final,
        mode="overwrite",
        storage_options=STORAGE_OPTIONS,
        engine="rust"
    )
    print("Pipeline de demanda finalizado com sucesso!")

# ==========================================
# DEFINIÇÃO DA DAG
# ==========================================
default_args = {
    'owner': 'engenharia_dados',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'Acougue_Previsao_Demanda_FLAML',
    default_args=default_args,
    description='Pipeline end-to-end de ML para previsão de demanda do Açougue',
    schedule_interval='0 2 * * *', # Roda todos os dias às 02:00 da manhã
    start_date=datetime(2026, 8, 23),
    catchup=False,
    tags=['machine_learning', 'demanda', 'acougue'],
) as dag:

    tarefa_pipeline_completo = PythonOperator(
        task_id='treinar_e_prever_demanda',
        python_callable=executar_pipeline_demanda,
    )
