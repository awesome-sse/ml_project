## ML in prodaction homework4

## Start server
### Install minikube on Windows:
    winget install minikube

### Start kubernetes cluster:
    minikube start --driver=docker

### *Start Docker Desktop*

### Copy docker image:
    minikube image load aweesomesse/ml_prod_homework2:v1 // minikube image load aweesomesse/ml_prod_homework2:v2

### Usage kuberctl:
    kubectl apply -f *filename*.yaml


### Info:
    kubectl get pods
    kubectl get replicaset
    kubectl get deployment

### Delete:
    kubectl delete --all pods
    kubectl delete --all replicaset
    kubectl delete --all deployment
