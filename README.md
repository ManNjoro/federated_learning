# FEDERATED LEARNING

Federated learning is a distributed learning approach that enables multiple parties to collaborate on a machine learning model
without sharing their data. This is particularly useful when data is sensitive or proprietary, and sharing it would
violate privacy or confidentiality agreements. In federated learning, each party trains a local model on their
own data and then shares the model updates with a central server. The central server aggregates the model updates
from all parties and updates the global model. This process is repeated until convergence.
Federated learning has several benefits, including:

* **Improved data privacy**: By not sharing data, federated learning helps protect sensitive information.
* **Increased data diversity**: By aggregating model updates from multiple parties, federated learning can
leverage diverse data sources and improve model performance.
* **Reduced communication overhead**: Federated learning reduces the need for data transmission, which can
be expensive and time-consuming.
* **Enhanced collaboration**: Federated learning enables multiple parties to collaborate on a model
without sharing their data, promoting cooperation and knowledge sharing.
Federated learning is particularly useful in applications where data is sensitive, such as:
* **Healthcare**: Federated learning can be used to develop models for disease diagnosis, patient
outcomes, and treatment recommendations without sharing patient data.
* **Finance**: Federated learning can be used to develop models for credit risk assessment, portfolio
management, and fraud detection without sharing financial data.
* **Government**: Federated learning can be used to develop models for public policy, resource allocation
and citizen services without sharing sensitive information.

## INSTALLATION

### Prequisites

------------

* node
* python

### STEP 1

```bash
git clone https://github.com/ManNjoro/federated_learning.git
```

### STEP 2

#### Create a virtual environment and activate it to install the required python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
```

##### Windows OS

```bash
python -m venv venv
venv\Scripts\activate
```

### STEP 3 - Installation of required dependencies

#### Frontend

* Navigate to frontend directory `cd frontend`
* Install the dependencies `npm i`
* Start the frontend server `npm run dev`

#### Backend

* Install the dependencies `pip install -r requirements.txt`
* Navigate to backend directory `cd backend`
* Start the backend server `python app.py`
