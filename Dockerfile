FROM apache/airflow:2.9.1-python3.10

# Muda temporariamente para root para instalar pacotes do sistema
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Volta para o usuário padrão do Airflow para instalar os pacotes Python
USER airflow
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
