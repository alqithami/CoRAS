FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
COPY requirements-cuda.txt requirements-cuda.txt
RUN python -m pip install --upgrade pip && pip install -r requirements-cuda.txt
COPY . /app
CMD ["bash"]
