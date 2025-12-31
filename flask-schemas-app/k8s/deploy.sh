#!/bin/bash

echo "Building Docker image in minikube..."
eval $(minikube docker-env)
docker build -t flask-schemas-app:latest ..

# Deployment script for Kubernetes manifests
# 
# This script applies Kubernetes configuration files to deploy the flask-schemas-app
# to a Kubernetes cluster.
#
# Usage:
#   ./deploy.sh
#
# Prerequisites:
#   - kubectl must be installed and configured
#   - Valid kubeconfig with access to target cluster
#   - Kubernetes manifest files must be present in the k8s directory
#
# Description:
#   Outputs a status message indicating that Kubernetes manifests are being applied.
#   This is typically followed by kubectl apply commands to deploy the application.
echo "Applying Kubernetes manifests..."
kubectl apply -f secret.yaml
kubectl apply -f configmap.yaml
kubectl apply -f mysql-pvc.yaml
kubectl apply -f mysql-deployment.yaml
kubectl apply -f mysql-service.yaml

echo "Waiting for MySQL to be ready..."
kubectl wait --for=condition=ready pod -l app=mysql --timeout=120s

echo "Deploying Flask application..."
kubectl apply -f flask-deployment.yaml
kubectl apply -f flask-service.yaml

echo "Waiting for Flask app to be ready..."
kubectl wait --for=condition=ready pod -l app=flask-app --timeout=120s

echo ""
echo "Deployment complete!"
echo ""
echo "To access the application, run:"
echo "  minikube service flask-service"
echo ""
echo "Or get the URL with:"
echo "  minikube service flask-service --url"
