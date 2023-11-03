# Use an official Python runtime as a parent image
FROM python:3.9.17

# copy the whole code 
COPY . /digits

# Install packages specified in requirements.txt
RUN pip install -r /digits/requirements.txt

# Set the working directory in the container
WORKDIR /digits

# Run exp.py when the container launches
CMD ["python", "exp.py"]

