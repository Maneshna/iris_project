# Iris Flower Prediction App

An end-to-end machine learning application built using:

- Scikit-learn
- FastAPI
- Streamlit

## Features

- Train Iris classification model
- Save model using Joblib
- FastAPI backend for predictions
- Streamlit frontend
- Client-server architecture

## Architecture

Streamlit Frontend
↓
FastAPI Backend
↓
Random Forest Model
↓
Prediction

## Run Backend

uvicorn api.main:app --reload

## Run Frontend

streamlit run frontend/app.py