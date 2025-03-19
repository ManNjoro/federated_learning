from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Expected columns in the dataset
EXPECTED_COLUMNS = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
    'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity', 'class'
]

# Load dataset (for demonstration, we'll use a placeholder)
data = pd.DataFrame()

# Preprocessing
def preprocess_data(data):
    # Validate columns
    if not all(column in data.columns for column in EXPECTED_COLUMNS):
        raise ValueError("Uploaded data does not have the expected columns.")
    
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data = data.replace({'Yes': 1, 'No': 0})
    features = data.drop(columns=['class'])
    labels = data['class'].map({'Positive': 1, 'Negative': 0})
    return features, labels

# Convert to TensorFlow datasets
def create_tf_dataset(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
    return dataset

# Define model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Convert to TFF model
def model_fn():
    keras_model = create_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=(tf.TensorSpec(shape=(None, 16), dtype=tf.float32),  # Input features
                   tf.TensorSpec(shape=(None, 1), dtype=tf.float32)),  # Labels
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# Federated averaging process
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

# Initialize the federated learning state
state = iterative_process.initialize()

# Endpoint to upload data
@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        # Load and preprocess the data
        global data
        data = pd.read_csv(filepath)
        features, labels = preprocess_data(data)
        return jsonify({'status': 'File uploaded and processed', 'filename': file.filename})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

# Endpoint to start federated training
@app.route('/start_training', methods=['POST'])
def start_training():
    global state, data
    if data.empty:
        return jsonify({'error': 'No data available for training'}), 400
    
    try:
        # Convert data to TensorFlow dataset
        features, labels = preprocess_data(data)
        client_dataset = create_tf_dataset(features, labels)
        
        # Simulate federated training
        for round_num in range(10):  # 10 rounds of federated training
            state, metrics = iterative_process.next(state, [client_dataset])
            print(f'Round {round_num}, Metrics: {metrics}')
        
        return jsonify({'status': 'Training completed', 'metrics': str(metrics)})
    except Exception as e:
        return jsonify({'error': f'Error during training: {str(e)}'}), 500

# Endpoint to get the current global model
@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    global state
    # Extract the global model weights
    global_weights = state.model_weights
    return jsonify({'weights': global_weights})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)