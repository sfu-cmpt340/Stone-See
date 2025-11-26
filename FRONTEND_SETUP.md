# Frontend Setup Guide

Complete guide to set up and run the KidneyNet frontend and backend.

## 🚀 Quick Start

### 1. Backend Setup (Flask API)

```bash
# Navigate to backend folder
cd backend

# Install dependencies (if using venv, activate it first)
pip install flask flask-cors

# Or install from requirements.txt
pip install -r requirements.txt

# Run the backend
python app.py
```

The backend will run at `http://localhost:5000`

### 2. Frontend Setup (Next.js)

```bash
# Navigate to frontend folder
cd frontend

# Install dependencies
npm install

# Run the frontend
npm run dev
```

The frontend will run at `http://localhost:3000`

## 📋 Prerequisites

- Python 3.8+ with Flask
- Node.js 16+ and npm
- Trained model in `results/models/` folder

## 🎯 Usage

1. **Start Backend**: Run `python backend/app.py`
2. **Start Frontend**: Run `npm run dev` in frontend folder
3. **Open Browser**: Go to `http://localhost:3000`
4. **Upload Image**: Click "Choose Image" and select a CT scan image
5. **Get Prediction**: Click "Analyze Image" to see results

## 🌐 Deploy to Vercel

### Frontend Deployment:

1. **Push to GitHub**:
   ```bash
   git add frontend/
   git commit -m "Add frontend"
   git push
   ```

2. **Deploy on Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Set root directory to `frontend`
   - Add environment variable:
     - Name: `NEXT_PUBLIC_API_URL`
     - Value: Your deployed backend URL (e.g., `https://your-backend.railway.app`)
   - Click "Deploy"

### Backend Deployment (Railway/Render):

#### Option 1: Railway
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select your repo
4. Set root directory to `backend`
5. Add environment variables if needed
6. Deploy!

#### Option 2: Render
1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
5. Deploy!

## 🔧 Troubleshooting

### Backend Issues:
- **Model not found**: Make sure you've trained the model first
- **Port already in use**: Change port in `app.py` (line with `app.run(port=5000)`)

### Frontend Issues:
- **Can't connect to backend**: Check `NEXT_PUBLIC_API_URL` in `.env.local`
- **Build errors**: Make sure Node.js version is 16+

### CORS Issues:
- Backend already has CORS enabled
- If issues persist, check backend is running

## 📝 Notes

- Backend must be running for frontend to work
- For production, update `NEXT_PUBLIC_API_URL` to your deployed backend URL
- The model loads automatically when backend starts

