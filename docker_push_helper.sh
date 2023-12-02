docker push sanjibmlops23.azurecr.io/dependency_digits
az acr build --image dependency_digits --registry sanjibmlops23 --file ./docker/DependencyDockerfile .

docker push sanjibmlops23.azurecr.io/digits:v1
az acr build --image digits:v1 --registry sanjibmlops23 --file ./docker/Dockerfile .