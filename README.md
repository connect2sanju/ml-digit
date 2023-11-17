## Flask execute 
export FLASK_APP=api/app.py
flask run 

## Docker setup
docker build -t assignment5:v3 -f docker/Dockerfile .
docker run -p 5000:5000 assignment5:v3


## Setting-up Azure-CLI
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update && brew install azure-cli
az --version


## Deployment 
git clone https://github.com/MicrosoftDocs/mslearn-deploy-run-container-app-service.git
cd mslearn-deploy-run-container-app-service/dotnet
az login --use-device 
az acr build --registry sanjibmlops23 --image webimage .

## Assignment-5
az acr build --image assignment5:v3 --registry sanjibmlops23 --file ./docker/Dockerfile .