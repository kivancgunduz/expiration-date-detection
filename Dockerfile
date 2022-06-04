FROM python:3.8
RUN mkdir /app
COPY . /app
WORKDIR /app
RUN apt libgl1
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
CMD python app.py