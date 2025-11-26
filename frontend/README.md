# KidneyNet Frontend

Frontend for Kidney Stone Detection System - Deployable on Vercel

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set your backend API URL in `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

3. Run development server:
```bash
npm run dev
```

4. Open http://localhost:3000 in your browser

## Deploy to Vercel

1. Push your code to GitHub
2. Go to [Vercel](https://vercel.com)
3. Import your repository
4. Set environment variable:
   - `NEXT_PUBLIC_API_URL` = your deployed backend URL
5. Deploy!

## Backend Setup

Make sure your Flask backend is running. See `../backend/README.md` for instructions.

