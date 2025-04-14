<<<<<<< HEAD
FROM python:3.12-slim
=======
FROM python:3.13-slim
>>>>>>> main

EXPOSE 8080
WORKDIR /app

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
