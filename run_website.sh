#!/bin/bash

# KidneyNet - Simple Run Script
# This script starts both backend and frontend servers

echo "=========================================="
echo "  KidneyNet - Starting Web Interface"
echo "=========================================="
echo ""

# Get the project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Kill any existing servers
echo "Stopping any existing servers..."
pkill -f "python.*app.py" 2>/dev/null
pkill -f "http.server" 2>/dev/null
sleep 2

# Start backend
echo "Starting backend API..."
cd backend
python3 app.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✓ Backend API running at http://localhost:5000"
else
    echo "✗ Backend failed to start. Check logs/backend.log"
    exit 1
fi

# Start frontend
echo "Starting frontend..."
cd frontend
python3 -m http.server 3000 > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 2

# Check if frontend is running
if curl -s http://localhost:3000 > /dev/null; then
    echo "✓ Frontend running at http://localhost:3000"
else
    echo "✗ Frontend failed to start. Check logs/frontend.log"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Open browser
echo ""
echo "Opening browser..."
open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000 in your browser"

echo ""
echo "=========================================="
echo "  ✓ Website is running!"
echo "=========================================="
echo ""
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop servers"
echo ""

# Wait for user to stop
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait

