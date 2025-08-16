FROM lauriewired/musicbeats:latest

# COPY . /remote-worker

# WORKDIR /remote-worker

# RUN pip install -e .

RUN pip install jupyter rq rq-scheduler


