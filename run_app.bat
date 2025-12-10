@echo off
echo Starting HumidityApp...

start "" http://localhost:8501

python\Scripts\python.exe -m streamlit run app\app.py

pause
