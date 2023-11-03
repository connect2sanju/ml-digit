# Use an official Python runtime as a parent image
FROM python:3.9.17

# copy the whole code 
COPY . /digits

# Install packages specified in requirements.txt
RUN pip install -r /digits/requirements.txt

# Set the working directory in the container
WORKDIR /digits

# Define a volume to save the trained models on the host machine
VOLUME /models

# Run exp.py when the container launches
# CMD ["python", "exp.py"]
CMD ["ls", "models"]