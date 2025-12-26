FROM python:3.12-slim

WORKDIR /app

# Copy everything so nano_backtest/ is available too
COPY . .

# Install deps if/when you add them
RUN pip install --no-cache-dir -r requirements.txt || true

# Default CSV path inside the container
ENV TICKS_CSV=/data/ticks.csv

# Run the backtester by default
ENTRYPOINT ["python", "backtest.py"]
