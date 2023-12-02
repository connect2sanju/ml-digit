# Use an official Python runtime as a parent image
# FROM python:3.9.17
FROM dependency_digits:latest

# copy the whole code 
COPY . /digits

# Install packages specified in requirements.txt
# RUN pip install -r /digits/requirements.txt

# Set the working directory in the container
WORKDIR /digits

# Set the FLASK_APP environment variable
ENV FLASK_APP api/new_app.py

EXPOSE 5000

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]

