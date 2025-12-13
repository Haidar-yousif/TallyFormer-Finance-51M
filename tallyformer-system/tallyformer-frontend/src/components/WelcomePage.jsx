import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const WelcomePage = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';    

    fetch(`${API_URL}/welcome`)
      .then(res => res.json())
      .then(data => {
        setModelInfo(data);
        setLoading(false);
      })
      .catch(() => {
        setError('Failed to load model info');
        setLoading(false);
      });
  }, []);

  return (
    <div className="welcome-container">
      <div className="welcome-card">
        {loading && <p style={{ textAlign: 'center', fontSize: '1.25rem' }}>Loading model info...</p>}
        {error && <p style={{ textAlign: 'center', color: '#f87171' }}>{error}</p>}
        {modelInfo && (
          <>
            <h1 className="welcome-title">Welcome to {modelInfo.model_name}</h1>
            <p className="welcome-message">{modelInfo.message}</p>
            <div className="stats-grid">
              {[
                { label: 'Model Size', value: modelInfo.model_size },
                { label: 'Context Length', value: modelInfo.context_length },
                { label: 'Layers', value: modelInfo.layers },
                { label: 'Attention Heads', value: modelInfo.heads },
                { label: 'Hidden Dim', value: modelInfo.hidden_dim },
              ].map((stat, idx) => (
                <div key={idx} className="stat-item">
                  <p className="stat-label">{stat.label}</p>
                  <p className="stat-value">{stat.value}</p>
                </div>
              ))}
            </div>
            <div style={{ textAlign: 'center' }}>
              
              <Link to="/chat" className="start-btn">
                Start Chatting
              </Link>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default WelcomePage;