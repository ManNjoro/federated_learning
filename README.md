# Federated Learning for Diabetes Detection

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-green)](https://nodejs.org/)

A privacy-preserving system that enables hospitals to collaboratively train diabetes prediction models without sharing patient data.

## Features

- 🛡️ **Privacy-first**: Data never leaves local hospitals
- 🤝 **Collaborative**: Multiple institutions improve model together
- 🏥 **Clinical UI**: Doctor-friendly dashboard with explanations
- ⚡ **Fast predictions**: <500ms inference via ONNX runtime

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+

### Installation

```bash
# Clone repo
git clone https://github.com/ManNjoro/federated_learning.git
cd federated_learning

# Set up Python
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Set up frontend
cd frontend
npm install
