# AI Futures Bot — live paper-trading container.
# Builds a lean image (pure-Python core). To use the ML strategies
# (ml_ensemble / ml_meta), uncomment the numpy/scikit-learn install below.
FROM python:3.11-slim

WORKDIR /app

# Core only needs PyYAML. Uncomment the second line for the ML strategies.
RUN pip install --no-cache-dir "PyYAML>=6.0"
# RUN pip install --no-cache-dir numpy scikit-learn

COPY ai_futures_bot ./ai_futures_bot
COPY examples ./examples

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
VOLUME ["/app/runtime"]

# Runs a standing $50k paper account and serves the dashboard on 0.0.0.0:8000.
# The account + state persist in the mounted /app/runtime volume.
CMD ["python", "-m", "ai_futures_bot.cli", "papertrade", \
     "--symbol", "MES", "--strategy", "ensemble", "--interval", "1", \
     "--serve", "--host", "0.0.0.0", "--port", "8000", \
     "--account", "runtime/account.json", "--state", "runtime/state.json"]
