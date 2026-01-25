#!/bin/bash

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
cd /app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend in the background
echo "Starting Streamlit frontend..."
cd /app/streamlit
streamlit run app.py --server.address=0.0.0.0 &

# Wait for any process to exit
wait -n
  
# Exit with status of process that exited first
exit $?
