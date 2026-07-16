#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Installation completed successfully!"
echo ""
echo "Run the application using:"
echo "streamlit run app.py"
