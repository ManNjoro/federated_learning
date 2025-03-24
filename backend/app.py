from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import pandas as pd
import os
import collections

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

def create_tf_dataset(features, labels):
    return tf.data.Dataset.from_tensor_slices((features.values, labels.values))

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

def model_fn():
    keras_model = create_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=collections.OrderedDict([
            ('x', tf.TensorSpec(shape=[None, 16], dtype=tf.float32)),
            ('y', tf.TensorSpec(shape=[None, 1], dtype=tf.float32))
        ]),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# Correct optimizer configuration for TFF 0.87.0
def client_optimizer_fn():
    return tf.keras.optimizers.Adam()

def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=1.0)

fed_avg = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=client_optimizer_fn,
    server_optimizer_fn=server_optimizer_fn
)

state = fed_avg.initialize()

# Flask endpoints remain the same as before
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
        global data
        data = pd.read_csv(filepath)
        features, labels = preprocess_data(data)
        return jsonify({'status': 'File uploaded and processed', 'filename': file.filename})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/start_training', methods=['POST'])
def start_training():
    global state, data
    if data.empty:
        return jsonify({'error': 'No data available for training'}), 400
    
    try:
        features, labels = preprocess_data(data)
        client_dataset = create_tf_dataset(features, labels)
        
        for round_num in range(10):
            result = fed_avg.next(state, [client_dataset])
            state = result.state
            metrics = result.metrics
            print(f'Round {round_num}, Metrics: {metrics}')
        
        return jsonify({'status': 'Training completed', 'metrics': str(metrics)})
    except Exception as e:
        return jsonify({'error': f'Error during training: {str(e)}'}), 500

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    global state
    return jsonify({'weights': state.model.trainable})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)