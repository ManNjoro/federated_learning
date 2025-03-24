from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple
from flwr.common import Metrics
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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
        x = self.sigmoid(self.fc3(x))
        return x

# Flower Client
class FlowerClient(fl.client.NumPyClient):
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

# Global model state
global_model = DiabetesModel()
global_train_loader = None
global_test_loader = None

# Metric aggregation
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples),
    }

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
    global global_model, global_train_loader, global_test_loader
    
    if global_train_loader is None:
        return jsonify({'error': 'No data available for training'}), 400
    
    try:
        # Start Flower server in a separate thread
        def start_server():
            strategy = fl.server.strategy.FedAvg(
                min_available_clients=1,
                evaluate_metrics_aggregation_fn=weighted_average,
                fit_metrics_aggregation_fn=weighted_average,
            )
            fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=fl.server.ServerConfig(num_rounds=3),
                strategy=strategy
            )
        
        import threading
        server_thread = threading.Thread(target=start_server)
        server_thread.start()
        
        # Start Flower client
        client = FlowerClient(global_model, global_train_loader, global_test_loader)
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
        
        return jsonify({'status': 'Training completed'})
    except Exception as e:
        return jsonify({'error': f'Error during training: {str(e)}'}), 500

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    global global_model
    weights = [param.detach().numpy().tolist() for param in global_model.parameters()]
    return jsonify({'weights': weights})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)