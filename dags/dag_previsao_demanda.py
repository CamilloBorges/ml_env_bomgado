from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator
from deltalake import DeltaTable, write_deltalake
from flaml import AutoML
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ==========================================
# CONFIGURAÇÕES E PARÂMETROS
# ==========================================
def get_airflow_setting(env_name: str, default: str | None = None, *, json_key: str | None = None):
    value = os.getenv(env_name)
    if value:
        return value

    try:
        connection = BaseHook.get_connection("Datalake")
    except AirflowException:
        connection = None

    if connection and json_key:
        extras = connection.extra_dejson or {}
        value = extras.get(json_key)
        if value:
            return value

    return default


LAKEHOUSE_URI_BRONZE = get_airflow_setting(
    "LAKEHOUSE_URI_BRONZE",
    "abfss://Bomgado@onelake.dfs.fabric.microsoft.com/armazem_lh.Lakehouse/Tables/dbo/bronze_treinamento",
    json_key="bronze_uri",
)
LAKEHOUSE_URI_PREVISAO = get_airflow_setting(
    "LAKEHOUSE_URI_PREVISAO",
    "abfss://Bomgado@onelake.dfs.fabric.microsoft.com/armazem_lh.Lakehouse/Tables/dbo/previsao_demanda_diaria_v2",
    json_key="previsao_uri",
)

FLAML_SETTINGS = {
    "time_budget": 180,
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

REQUIRED_COLUMNS = {"ds", "eanBalanca", "quantidade"}


class FlamlForecastWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, automl):
        self.automl = automl

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            return self.automl.predict(model_input)
        return self.automl.predict(pd.DataFrame(model_input))


def build_storage_options():
    try:
        connection = BaseHook.get_connection("Datalake")
    except AirflowException:
        connection = None

    tenant_id = os.getenv("AZURE_TENANT_ID") or (
        connection.extra_dejson.get("tenantId") if connection and connection.extra_dejson else None
    )
    client_id = os.getenv("AZURE_CLIENT_ID") or (connection.login if connection else None)
    client_secret = os.getenv("AZURE_CLIENT_SECRET") or (connection.password if connection else None)

    if not all([tenant_id, client_id, client_secret]):
        raise ValueError(
            "Credenciais do Lakehouse ausentes. Configure AZURE_TENANT_ID, AZURE_CLIENT_ID e AZURE_CLIENT_SECRET, "
            "ou cadastre a conexão 'Datalake' no Airflow."
        )

    return {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
    }


def get_mlflow_tracking_uri():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri

    candidate_paths = [
        "/mlruns/mlflow.db",
        os.path.join(os.getcwd(), "mlruns", "mlflow.db"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return f"sqlite:///{path}"

    return "http://mlflow-server:5000"


# ==========================================
# FUNÇÕES DE PREPROCESSAMENTO
# ==========================================
def _coerce_na_for_sklearn(df):
    df = df.copy()
    for col in df.columns:
        s = df[col]
        dt = s.dtype
        is_object_like = (
            isinstance(dt, pd.CategoricalDtype)
            or dt == object
            or dt == "boolean"
            or str(dt) == "string"
        )
        if is_object_like:
            s_obj = s.astype(object)
            df[col] = s_obj.where(s_obj.notna(), np.nan)
    return df


def create_fillna_processor(df):
    mean_features = [
        col
        for col in df.select_dtypes(include=["number"], exclude=["timedelta"]).columns
        if df[col].skew(skipna=True) <= 1
    ]
    median_features = [
        col
        for col in df.select_dtypes(include=["number"], exclude=["timedelta"]).columns
        if df[col].skew(skipna=True) > 1
    ]

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


def validate_history_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("O histórico do Lakehouse está vazio.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame sem colunas obrigatórias: {missing_str}")

    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df[df["ds"].notna()].copy()
    if df.empty:
        raise ValueError("Nenhuma data válida encontrada na coluna ds.")

    df["eanBalanca"] = df["eanBalanca"].astype(str)
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce")
    df = df[df["quantidade"].notnull()].copy()
    if df.empty:
        raise ValueError("Nenhuma linha com quantidade válida foi encontrada para treino.")

    df = df[df["ds"] >= "2024-01-01"].copy()
    if df.empty:
        raise ValueError("Sem dados históricos válidos a partir de 2024-01-01.")

    df = df.rename(columns=lambda c: re.sub(r"[^A-Za-z0-9_]+", "_", c))
    return df


def carregar_dados_validos():
    storage_options = build_storage_options()
    dt = DeltaTable(LAKEHOUSE_URI_BRONZE, storage_options=storage_options)
    df = dt.to_pandas()
    validated = validate_history_dataframe(df)
    print(f"Linhas carregadas para treino: {len(validated)}")
    return validated


# ==========================================
# PIPELINE PRINCIPAL DO AIRFLOW
# ==========================================
def executar_pipeline_demanda(df: pd.DataFrame | None = None):
    storage_options = build_storage_options()
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("IA_Previsao_Demanda_Acougue")

    if df is None:
        print("[Etapa 0] Carregando e validando histórico do Lakehouse...")
        df = carregar_dados_validos()
    else:
        df = validate_history_dataframe(df)
        print(f"[Etapa 0] Histórico recebido via XCom com {len(df)} linhas.")

    print("[Etapa 1] Preparando dados de treino...")
    top_skus = df["eanBalanca"].value_counts().head(50).index
    df_hist = df[df["eanBalanca"].isin(top_skus)].copy()
    df_hist["eanBalanca"] = df_hist["eanBalanca"].astype("category")
    time_col = "ds"
    item_col = "eanBalanca"
    target_col = "quantidade"

    X = df_hist.select_dtypes(include=["number", "datetime", "category", "boolean"], exclude=["timedelta"])
    X_sorted = X.sort_values(time_col)
    unique_dates = np.sort(X_sorted[time_col].unique())
    if len(unique_dates) < 2:
        raise ValueError("Quantidade insuficiente de datas para treinar o modelo.")

    cutoff_date = unique_dates[int(len(unique_dates) * 0.8)]
    X_train = X_sorted[X_sorted[time_col] <= cutoff_date].copy()
    y_train = X_train.pop(target_col)

    preprocessor, all_features, datetime_features = create_fillna_processor(X_train)
    X_train_filled = fillna(X_train, preprocessor, all_features, datetime_features)

    Xy_train = X_train_filled.copy()
    Xy_train[target_col] = y_train
    print("[Etapa 2] Resample diário por SKU...")
    Xy_resampled = Xy_train.groupby(item_col, group_keys=False).apply(
        lambda g: resample_item(g, time_col, target_col)
    ).reset_index(drop=True)

    print("[Etapa 3] Iniciando AutoML (FLAML)...")
    automl = AutoML()
    with mlflow.start_run(run_name="demanda_acougue_treino") as run:
        automl.fit(
            dataframe=Xy_resampled,
            label=target_col,
            period=25,
            time_col=time_col,
            group_id=item_col,
            **FLAML_SETTINGS,
        )

        mlflow.pyfunc.log_model(
            python_model=FlamlForecastWrapper(automl),
            artifact_path="model",
            registered_model_name="IA_Previsao_Demanda_Acougue",
            input_example=Xy_resampled.head(5),
        )
        print(f"Modelo registrado com sucesso. Run ID: {run.info.run_id}")

    print("[Etapa 4] Gerando previsões futuras...")
    df_fut = df[df[time_col] > pd.Timestamp.now()].copy()
    if df_fut.empty:
        raise ValueError("Nenhuma data futura encontrada em bronze_treinamento para inferência.")

    df_fut[time_col] = pd.to_datetime(df_fut[time_col])
    df_fut[item_col] = df_fut[item_col].astype("category")

    expected_cols = list(all_features) + list(datetime_features)
    for col_name in expected_cols:
        if col_name not in df_fut.columns:
            df_fut[col_name] = np.nan

    X_fut_for_imputer = df_fut[expected_cols]
    X_fut_full = fillna(X_fut_for_imputer, preprocessor, all_features, datetime_features)
    feature_cols = list(X_train.columns)
    X_fut_features = X_fut_full.reindex(columns=feature_cols, fill_value=np.nan)

    previsoes = automl.predict(X_fut_features)
    df_pred_final = pd.DataFrame(
        {
            "ds": X_fut_full[time_col].values,
            "eanBalanca": X_fut_full[item_col].astype(str).values,
            "quantidade_prevista": previsoes,
        }
    )

    print("[Etapa 5] Gravando previsões de volta no Lakehouse (Delta)...")
    write_deltalake(
        LAKEHOUSE_URI_PREVISAO,
        df_pred_final,
        mode="overwrite",
        storage_options=storage_options,
        engine="rust",
    )
    print("Pipeline de demanda finalizado com sucesso!")
    return df_pred_final


# ==========================================
# DEFINIÇÃO DA DAG
# ==========================================
default_args = {
    "owner": "engenharia_dados",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "Acougue_Previsao_Demanda_FLAML",
    default_args=default_args,
    description="Pipeline end-to-end de ML para previsão de demanda do Açougue",
    schedule_interval="0 2 * * *",
    start_date=datetime(2026, 8, 23),
    catchup=False,
    tags=["machine_learning", "demanda", "acougue"],
) as dag:

    carregar_dados = PythonOperator(
        task_id="carregar_dados_validos",
        python_callable=carregar_dados_validos,
    )

    treinar_e_prever = PythonOperator(
        task_id="treinar_e_prever_demanda",
        python_callable=executar_pipeline_demanda,
    )

    carregar_dados >> treinar_e_prever
