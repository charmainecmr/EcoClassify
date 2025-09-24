@echo off
echo === Setting up EcoClassify Virtual Environment ===

:: Create virtual environment
python -m venv venv

:: Activate environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
pip install -r requirements.txt

echo === Setup Complete! ===
echo To start your app, run:
echo     venv\Scripts\activate
echo     streamlit run streamlit_app.py
