FROM lauriewired/musicbeats:latest

# COPY . /remote-worker

# WORKDIR /remote-worker

# RUN pip install -e .

RUN pip install fastapi uvicorn

WORKDIR /app

COPY cache/ /root/.cache/

COPY main.py .

ENTRYPOINT [  ]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


