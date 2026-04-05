@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
.\venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo Setup complete! To activate the environment in future sessions, run: .\venv\Scripts\activate
pause