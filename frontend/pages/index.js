import { useState } from 'react';
import Head from 'next/head';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Backend API URL - change this to your deployed backend URL
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <>
      <Head>
        <title>KidneyNet - Kidney Stone Detection</title>
        <meta name="description" content="Detect kidney stones in CT scan images using AI" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="container">
        <header>
          <h1>🫘 KidneyNet</h1>
          <p>Kidney Stone Detection System</p>
        </header>

        <main>
          <div className="upload-section">
            <h2>Upload CT Scan Image</h2>
            
            {!preview ? (
              <div className="upload-area">
                <input
                  type="file"
                  id="file-input"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
                <label htmlFor="file-input" className="upload-button">
                  📁 Choose Image
                </label>
                <p className="upload-hint">Supported formats: JPG, PNG, JPEG</p>
              </div>
            ) : (
              <div className="preview-section">
                <div className="image-preview">
                  <img src={preview} alt="Preview" />
                  <button onClick={handleReset} className="reset-button">
                    ✕ Remove
                  </button>
                </div>
                
                <button
                  onClick={handlePredict}
                  disabled={loading}
                  className="predict-button"
                >
                  {loading ? '⏳ Analyzing...' : '🔍 Analyze Image'}
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          {result && (
            <div className="result-section">
              <h2>Results</h2>
              
              <div className={`prediction-card ${result.prediction === 'Stone' ? 'stone' : 'non-stone'}`}>
                <div className="prediction-header">
                  <h3>
                    {result.prediction === 'Stone' ? '🟡 Kidney Stone Detected' : '✅ No Kidney Stone'}
                  </h3>
                  <div className="confidence">
                    Confidence: {result.confidence}%
                  </div>
                </div>
                
                <div className="probabilities">
                  <div className="prob-item">
                    <span className="prob-label">Non-Stone:</span>
                    <div className="prob-bar">
                      <div 
                        className="prob-fill non-stone"
                        style={{ width: `${result.probabilities['Non-Stone']}%` }}
                      ></div>
                      <span className="prob-value">{result.probabilities['Non-Stone']}%</span>
                    </div>
                  </div>
                  
                  <div className="prob-item">
                    <span className="prob-label">Stone:</span>
                    <div className="prob-bar">
                      <div 
                        className="prob-fill stone"
                        style={{ width: `${result.probabilities['Stone']}%` }}
                      ></div>
                      <span className="prob-value">{result.probabilities['Stone']}%</span>
                    </div>
                  </div>
                </div>
              </div>

              <button onClick={handleReset} className="new-analysis-button">
                🔄 Analyze Another Image
              </button>
            </div>
          )}
        </main>

        <footer>
          <p>KidneyNet - AI-Powered Kidney Stone Detection</p>
          <p className="disclaimer">
            ⚠️ This tool is for research purposes only. Not a substitute for professional medical diagnosis.
          </p>
        </footer>
      </div>

      <style jsx>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        .container {
          min-height: 100vh;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }

        header {
          text-align: center;
          color: white;
          margin-bottom: 40px;
          padding-top: 20px;
        }

        header h1 {
          font-size: 3rem;
          margin-bottom: 10px;
          text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        header p {
          font-size: 1.2rem;
          opacity: 0.9;
        }

        main {
          max-width: 800px;
          margin: 0 auto;
          background: white;
          border-radius: 20px;
          padding: 40px;
          box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }

        .upload-section h2 {
          margin-bottom: 20px;
          color: #333;
        }

        .upload-area {
          text-align: center;
          padding: 60px 20px;
          border: 3px dashed #667eea;
          border-radius: 15px;
          background: #f8f9ff;
        }

        .upload-button {
          display: inline-block;
          padding: 15px 40px;
          background: #667eea;
          color: white;
          border-radius: 10px;
          cursor: pointer;
          font-size: 1.1rem;
          font-weight: 600;
          transition: all 0.3s;
        }

        .upload-button:hover {
          background: #5568d3;
          transform: translateY(-2px);
          box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .upload-hint {
          margin-top: 15px;
          color: #666;
          font-size: 0.9rem;
        }

        .preview-section {
          text-align: center;
        }

        .image-preview {
          position: relative;
          display: inline-block;
          margin-bottom: 20px;
        }

        .image-preview img {
          max-width: 100%;
          max-height: 400px;
          border-radius: 10px;
          box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .reset-button {
          position: absolute;
          top: 10px;
          right: 10px;
          background: rgba(255, 0, 0, 0.8);
          color: white;
          border: none;
          border-radius: 50%;
          width: 35px;
          height: 35px;
          cursor: pointer;
          font-size: 1.2rem;
          transition: all 0.3s;
        }

        .reset-button:hover {
          background: rgba(255, 0, 0, 1);
          transform: scale(1.1);
        }

        .predict-button {
          padding: 15px 40px;
          background: #667eea;
          color: white;
          border: none;
          border-radius: 10px;
          font-size: 1.1rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s;
        }

        .predict-button:hover:not(:disabled) {
          background: #5568d3;
          transform: translateY(-2px);
          box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .predict-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .error-message {
          background: #fee;
          color: #c33;
          padding: 15px;
          border-radius: 10px;
          margin-top: 20px;
          border-left: 4px solid #c33;
        }

        .result-section {
          margin-top: 30px;
        }

        .result-section h2 {
          margin-bottom: 20px;
          color: #333;
        }

        .prediction-card {
          padding: 30px;
          border-radius: 15px;
          margin-bottom: 20px;
          box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .prediction-card.stone {
          background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
          border-left: 5px solid #ffc107;
        }

        .prediction-card.non-stone {
          background: linear-gradient(135deg, #d1f2eb 0%, #a3e4d7 100%);
          border-left: 5px solid #28a745;
        }

        .prediction-header {
          margin-bottom: 20px;
        }

        .prediction-header h3 {
          font-size: 1.5rem;
          margin-bottom: 10px;
          color: #333;
        }

        .confidence {
          font-size: 1.1rem;
          font-weight: 600;
          color: #555;
        }

        .probabilities {
          margin-top: 20px;
        }

        .prob-item {
          margin-bottom: 15px;
        }

        .prob-label {
          display: block;
          margin-bottom: 5px;
          font-weight: 600;
          color: #333;
        }

        .prob-bar {
          position: relative;
          background: #e0e0e0;
          border-radius: 10px;
          height: 30px;
          overflow: hidden;
        }

        .prob-fill {
          height: 100%;
          border-radius: 10px;
          transition: width 0.5s;
          display: flex;
          align-items: center;
          justify-content: flex-end;
          padding-right: 10px;
          color: white;
          font-weight: 600;
        }

        .prob-fill.non-stone {
          background: linear-gradient(90deg, #28a745, #20c997);
        }

        .prob-fill.stone {
          background: linear-gradient(90deg, #ffc107, #ff9800);
        }

        .prob-value {
          position: absolute;
          right: 10px;
          top: 50%;
          transform: translateY(-50%);
          font-weight: 600;
          color: #333;
        }

        .new-analysis-button {
          width: 100%;
          padding: 15px;
          background: #667eea;
          color: white;
          border: none;
          border-radius: 10px;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s;
        }

        .new-analysis-button:hover {
          background: #5568d3;
          transform: translateY(-2px);
          box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        footer {
          text-align: center;
          color: white;
          margin-top: 40px;
          padding: 20px;
        }

        footer p {
          margin: 5px 0;
        }

        .disclaimer {
          font-size: 0.9rem;
          opacity: 0.8;
          margin-top: 10px;
        }

        @media (max-width: 768px) {
          header h1 {
            font-size: 2rem;
          }

          main {
            padding: 20px;
          }

          .upload-area {
            padding: 40px 15px;
          }
        }
      `}</style>
    </>
  );
}

