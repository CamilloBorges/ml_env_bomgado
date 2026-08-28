import sys
import types

import pandas as pd
import pytest


airflow_module = types.ModuleType("airflow")
exceptions_module = types.ModuleType("airflow.exceptions")
hooks_module = types.ModuleType("airflow.hooks")
hooks_base_module = types.ModuleType("airflow.hooks.base")
operators_module = types.ModuleType("airflow.operators")
python_module = types.ModuleType("airflow.operators.python")


class AirflowException(Exception):
    pass


class BaseHook:
    @staticmethod
    def get_connection(_name):
        raise AirflowException("connection unavailable")


class PythonOperator:
    def __init__(self, *args, **kwargs):
        self.task_id = kwargs.get("task_id")

    def __rshift__(self, other):
        return other


class DAG:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


exceptions_module.AirflowException = AirflowException
hooks_base_module.BaseHook = BaseHook
python_module.PythonOperator = PythonOperator
airflow_module.DAG = DAG
airflow_module.exceptions = exceptions_module
airflow_module.hooks = hooks_module
airflow_module.operators = operators_module
hooks_module.base = hooks_base_module
operators_module.python = python_module

sys.modules.setdefault("airflow", airflow_module)
sys.modules.setdefault("airflow.exceptions", exceptions_module)
sys.modules.setdefault("airflow.hooks", hooks_module)
sys.modules.setdefault("airflow.hooks.base", hooks_base_module)
sys.modules.setdefault("airflow.operators", operators_module)
sys.modules.setdefault("airflow.operators.python", python_module)

mlflow_module = types.ModuleType("mlflow")
pyfunc_module = types.ModuleType("mlflow.pyfunc")


class PythonModel:
    pass


class _RunContext:
    def __init__(self, run_id):
        self.info = types.SimpleNamespace(run_id=run_id)


class _MLflowMock:
    @staticmethod
    def set_tracking_uri(_uri):
        return None

    @staticmethod
    def set_experiment(_name):
        return None

    @staticmethod
    def start_run(*args, **kwargs):
        return _RunContext("mock-run")

    @staticmethod
    def pyfunc_log_model(*args, **kwargs):
        return None

    @staticmethod
    def sklearn_log_model(*args, **kwargs):
        return None


mlflow_module.pyfunc = pyfunc_module
pyfunc_module.PythonModel = PythonModel
mlflow_module.set_tracking_uri = _MLflowMock.set_tracking_uri
mlflow_module.set_experiment = _MLflowMock.set_experiment
mlflow_module.start_run = _MLflowMock.start_run
mlflow_module.pyfunc = types.SimpleNamespace(log_model=_MLflowMock.pyfunc_log_model, PythonModel=PythonModel)
mlflow_module.sklearn = types.SimpleNamespace(log_model=_MLflowMock.sklearn_log_model)

sys.modules.setdefault("mlflow", mlflow_module)
sys.modules.setdefault("mlflow.pyfunc", pyfunc_module)

flaml_module = types.ModuleType("flaml")
flaml_module.AutoML = type("AutoML", (), {})
sys.modules.setdefault("flaml", flaml_module)

deltalake_module = types.ModuleType("deltalake")
deltalake_module.DeltaTable = type("DeltaTable", (), {})
deltalake_module.write_deltalake = lambda *args, **kwargs: None
sys.modules.setdefault("deltalake", deltalake_module)

from dags.dag_previsao_demanda import (
    build_storage_options,
    get_airflow_setting,
    validate_history_dataframe,
)


def test_validate_history_dataframe_keeps_expected_columns():
    df = pd.DataFrame(
        {
            "ds": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "eanBalanca": ["100", "100", "200"],
            "quantidade": [10, 12, 8],
        }
    )

    result = validate_history_dataframe(df)

    assert list(result.columns) == ["ds", "eanBalanca", "quantidade"]
    assert result["ds"].dtype.kind == "M"
    assert result["quantidade"].dtype.kind in {"i", "f"}


def test_validate_history_dataframe_rejects_missing_columns():
    df = pd.DataFrame({"ds": ["2024-01-01"], "quantidade": [10]})

    with pytest.raises(ValueError, match="eanBalanca"):
        validate_history_dataframe(df)


def test_build_storage_options_uses_environment(monkeypatch):
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-123")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-123")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret-123")

    storage_options = build_storage_options()

    assert storage_options["tenant_id"] == "tenant-123"
    assert storage_options["client_id"] == "client-123"
    assert storage_options["client_secret"] == "secret-123"


def test_get_airflow_setting_prefers_environment(monkeypatch):
    monkeypatch.setenv("LAKEHOUSE_URI_BRONZE", "env-bronze-uri")
    monkeypatch.delenv("AIRFLOW__CORE__LOAD_EXAMPLES", raising=False)

    value = get_airflow_setting("LAKEHOUSE_URI_BRONZE", "default-bronze-uri")

    assert value == "env-bronze-uri"
