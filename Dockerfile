FROM python:3.7-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc wget

# Install pytorch and fastai
# RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai

# Install starlette and uvicorn
RUN pip install starlette uvicorn python-multipart aiohttp

ADD run.py run.py

RUN wget -q https://storage.googleapis.com/model-dist-112/models/model.pth && mkdir -p data/models && mv model.pth data/models/
# RUN mkdir -p data/models
# COPY model.pth data/models/
# RUN mkdir -p data/models && mv model.pth data/models/
EXPOSE 8008

# Start the server
CMD ["python", "run.py", "serve"]
