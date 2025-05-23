FROM python:3.12.9-slim
WORKDIR /root

# Install git
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /root/
RUN pip install -r requirements.txt
COPY app.py /root/
ENTRYPOINT ["python"]
CMD ["app.py"]

