FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 0.0.0.0 listens on every container interface, allowing Docker's port
# mapping to reach Uvicorn. 127.0.0.1 would accept only in-container traffic.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
