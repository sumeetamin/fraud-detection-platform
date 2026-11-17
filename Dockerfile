FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN version="$(tr -d '\r\n' < artifacts/current.txt)" && if [ -n "$version" ] && [ -f "artifacts/${version}.joblib" ]; then echo "Using supplied model artifact ${version}"; else echo "No matching model artifact supplied; training the labeled synthetic demo model" && python -m fraud.train; fi
EXPOSE 8501 8003
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
