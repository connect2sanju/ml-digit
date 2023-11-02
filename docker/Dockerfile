FROM python:3.9.17

# copy the whole code 
COPY . /digits

RUN pip install -r /digits/requirements.txt

WORKDIR /digits

CMD ["pytest"]
