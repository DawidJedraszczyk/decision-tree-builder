FROM python:3.9-slim

RUN apt-get update && apt-get install -y graphviz xdg-utils

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED 1

# Run the decision tree program
CMD ["python", "main.py"]
