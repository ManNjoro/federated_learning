from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple, Optional
from flwr.common import Metrics
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import subprocess
import threading
import time

app = Flask(__name__)

# Directory setup
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Expected dataset columns
EXPECTED_COLUMNS = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
    'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'class'
]

# Initialize empty dataframe
data = pd.DataFrame()

# Data preprocessing
def preprocess_data(data):
    if not all(column in data.columns for column in EXPECTED_COLUMNS):
        raise ValueError("Uploaded data does not have the expected columns.")
    
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data = data.replace({'Yes': 1, 'No': 0})
    features = data.drop(columns=['class'])
    labels = data['class'].map({'Positive': 1, 'Negative': 0})
    return features, labels

# Neural Network Model
class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

# Flower Client
class FlowerClient(fl.client.Client):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    def get_parameters(self, config):
        return [param.detach().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(3):  # 3 local epochs
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": loss.item()}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                outputs = self.model(X_batch)
                total_loss += self.criterion(outputs, y_batch).item()
                predicted = (outputs >= 0.5).float()
                correct += (predicted == y_batch).sum().item()
        accuracy = correct / len(self.test_loader.dataset)
        return float(total_loss / len(self.test_loader)), len(self.test_loader.dataset), {"accuracy": accuracy}

# Global state
global_model = DiabetesModel()
global_train_loader = None
global_test_loader = None
server_process = None
client_process = None

# Metric aggregation
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples),
    }

def start_superlink():
    """Start the Flower SuperLink in a subprocess"""
    global server_process
    server_process = subprocess.Popen([
        "flower-superlink",
        "--insecure",
        "--min-available-clients=1",
        "--port=8080"
    ])

def start_supernode():
    """Start the Flower client as a SuperNode"""
    global client_process, global_model, global_train_loader, global_test_loader
    if global_train_loader is None:
        return False
    
    client = FlowerClient(global_model, global_train_loader, global_test_loader)
    client_process = subprocess.Popen([
        "flower-supernode",
        "--insecure",
        "--superlink=localhost:8080",
        f"--client={client.__module__}:{client.__class__.__name__}"
    ])
    return True

# Flask endpoints
@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        global data, global_train_loader, global_test_loader
        data = pd.read_csv(filepath)
        features, labels = preprocess_data(data)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features.values, labels.values, test_size=0.2
        )
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        )
        
        global_train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        global_test_loader = DataLoader(test_dataset, batch_size=16)
        
        return jsonify({'status': 'File uploaded and processed', 'filename': file.filename})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/start_training', methods=['POST'])
def start_training():
    global server_process, client_process
    
    # Start SuperLink if not running
    if server_process is None:
        start_superlink()
        time.sleep(2)  # Give server time to start
    
    # Start SuperNode
    if not start_supernode():
        return jsonify({'error': 'No data available for training'}), 400
    
    return jsonify({'status': 'Training started'})

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    global global_model
    weights = [param.detach().numpy().tolist() for param in global_model.parameters()]
    return jsonify({'weights': weights})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global server_process, client_process
    if client_process:
        client_process.terminate()
    if server_process:
        server_process.terminate()
    return jsonify({'status': 'Training stopped'})

# Add this endpoint with the others in your Flask app
@app.route('/predict', methods=['POST'])
def predict():
    global global_model
    
    try:
        # 1. Get input data from request
        input_data = request.json
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # 2. Convert input to DataFrame for preprocessing
        input_df = pd.DataFrame([input_data])
        
        # 3. Preprocess (same as training)
        input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
        input_df = input_df.replace({'Yes': 1, 'No': 0})
        
        # 4. Validate all 16 features are present
        missing_cols = set(EXPECTED_COLUMNS) - set(input_df.columns) - {'class'}
        if missing_cols:
            return jsonify({'error': f'Missing features: {missing_cols}'}), 400
        
        # 5. Prepare tensor
        features = input_df[EXPECTED_COLUMNS[:-1]]  # exclude 'class'
        input_tensor = torch.tensor(features.values.astype(float), dtype=torch.float32)
        
        # 6. Make prediction
        with torch.no_grad():
            global_model.eval()
            prediction = global_model(input_tensor).item()  # sigmoid output 0-1
        
        # 7. Format response
        risk_level = "High" if prediction >= 0.7 else "Medium" if prediction >= 0.3 else "Low"
        
        return jsonify({
            'prediction': prediction,
            'risk_level': risk_level,
            'interpretation': f"{risk_level} risk of diabetes",
            'important_features': get_top_features(input_tensor)  # See helper function below
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper function to explain predictions
def get_top_features(input_tensor):
    """Returns the 3 most influential features for the prediction"""
    global global_model
    
    # 1. Get gradients to see feature importance
    input_tensor.requires_grad = True
    global_model.zero_grad()
    prediction = global_model(input_tensor)
    prediction.backward()
    
    # 2. Get absolute gradient values
    grads = input_tensor.grad.data.abs().numpy()[0]
    feature_importance = dict(zip(EXPECTED_COLUMNS[:-1], grads))
    
    # 3. Return top 3 influential features
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    return [{
        'feature': feature,
        'importance': float(importance),
        'direction': "Positive" if input_tensor[0][i].item() > 0 else "Negative"
    } for i, (feature, importance) in enumerate(top_features)]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    