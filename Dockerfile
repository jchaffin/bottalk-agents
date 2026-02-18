FROM dailyco/pipecat-base:latest

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./bot.py bot.py
COPY ./agent.py agent.py
COPY ./latency.py latency.py
COPY ./sarah.py sarah.py
COPY ./mike.py mike.py
