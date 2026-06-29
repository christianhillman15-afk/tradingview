# AI Futures Bot — live paper-trading container.
# The core, dashboard, and papertrade run on the Python STANDARD LIBRARY only,
# so this image needs no pip install and builds offline. Optional extras:
#   - PyYAML        -> only if you pass a YAML --config file
#   - numpy/sklearn -> only for the ML strategies (ml_ensemble / ml_meta)
# Uncomment the RUN line to add them.
FROM python:3.11-slim

WORKDIR /app

# RUN pip install --no-cache-dir "PyYAML>=6.0" numpy scikit-learn

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
