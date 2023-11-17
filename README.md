#Flask execute 
export FLASK_APP=api/hello-world.py
flask run 

#Docker setup
docker build -t flask:v1 -f docker/Dockerfile .
docker run -p 5000:5000 flask:v1