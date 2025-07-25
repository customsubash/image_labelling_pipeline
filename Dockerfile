FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

EXPOSE 7000

CMD ["python", "main.py"]
