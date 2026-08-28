from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from deltalake import DeltaTable, write_deltalake
import os

def run_training_pipeline():
    # 1. Leitura do Delta diretamente sem Spark
    storage_options = {
        "tenant_id": os.environ.get("AZURE_TENANT_ID"),
        "client_id": os.environ.get("AZURE_CLIENT_ID"),
        "client_secret": os.environ.get("AZURE_CLIENT_SECRET")
    }
    
    # Caminho do lakehouse (ajuste para a sua URL do ADLS/OneLake)
    tabela_bronze_uri = "abfs://seu_container@sua_conta.dfs.core.windows.net/Tables/dbo/bronze_treinamento"
    
    print("[Step 1] Carregando dados via deltalake...")
    dt = DeltaTable(tabela_bronze_uri, storage_options=storage_options)
    
    # Converte para Pandas (já em memória) e filtra valores nulos
    df = dt.to_pandas()
    df_hist = df[df['quantidade'].notnull()].copy()
    
    # Daqui em diante, seu código original em Pandas continua exatamente igual:
    # - Split temporal
    # - FLAML AutoML (automl.fit)
    # - Registro no MLflow (apontando para http://mlflow-server:5000)
    
    print("Treinamento finalizado!")

def run_predictions():
    # Mesma lógica do step final: ler deltalake, filtrar datas futuras, prever e salvar
    # write_deltalake(tabela_previsao_uri, df_pred_batch, mode="overwrite", storage_options=storage_options)
    pass

with DAG("Acougue_Previsao_Demanda", start_date=datetime(2026, 8, 24), schedule_interval="@daily", catchup=False) as dag:
    
    tarefa_treino = PythonOperator(
        task_id="treinar_modelo",
        python_callable=run_training_pipeline
    )
    
    tarefa_previsao = PythonOperator(
        task_id="gerar_previsoes_diarias",
        python_callable=run_predictions
    )

    tarefa_treino >> tarefa_previsao
