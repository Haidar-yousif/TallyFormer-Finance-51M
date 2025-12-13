import React, { useState, useContext } from 'react';
import { Copy, Edit2, RotateCcw, Check } from 'lucide-react';
import toast from 'react-hot-toast';
import { ThemeContext } from '../App';
import darkIcon from '../assets/icon-dark-514-514.jpg';
import lightIcon from '../assets/icon-light-514-514.jpg';

const MessageBubble = ({ message, onRegenerate, onEditSave, index }) => {
  const { theme } = useContext(ThemeContext);
  const { prompt, response, metrics, isLoading } = message;
  const [isEditing, setIsEditing] = useState(false);
  const [editedPrompt, setEditedPrompt] = useState(prompt);
  const [copiedPrompt, setCopiedPrompt] = useState(false);
  const [copiedResponse, setCopiedResponse] = useState(false);
  const isDark = theme === 'dark';

  const copyText = (text, setter) => {
    navigator.clipboard.writeText(text);
    setter(true);
    toast.success('Copied!', { duration: 1500 });
    setTimeout(() => setter(false), 1500);
  };

  const handleEdit = () => setIsEditing(true);
  const handleSave = () => {
    onEditSave(index, editedPrompt.trim());
    setIsEditing(false);
  };
  const handleCancel = () => {
    setIsEditing(false);
    setEditedPrompt(prompt);
  };

  return (
    <div className="message-group">
      {/* User Prompt */}
      <div className="user-bubble">
        <div className={`user-bubble-inner ${isDark ? 'dark' : 'light'}`}>
          <div className="bubble-header">
            <p>You</p>
            <div className="bubble-actions">
              {!isEditing && (
                <>
                  <button onClick={() => copyText(prompt, setCopiedPrompt)} className="action-btn">
                    {copiedPrompt ? <Check size={16} /> : <Copy size={16} />}
                  </button>
                  <button onClick={handleEdit} className="action-btn">
                    <Edit2 size={16} />
                  </button>
                </>
              )}
            </div>
          </div>

          {isEditing ? (
            <div>
              <textarea
                value={editedPrompt}
                onChange={(e) => setEditedPrompt(e.target.value)}
                className={`textarea-input ${isDark ? 'dark' : 'light'}`}
                rows="4"
                autoFocus
              />
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', marginTop: '0.75rem' }}>
                <button onClick={handleCancel}  className="edit-btn cancel">
                  Cancel
                </button>
                <button onClick={handleSave} className="edit-btn save">
                  Save & Submit
                </button>
              </div>
            </div>
          ) : (
            <p style={{ whiteSpace: 'pre-wrap' }}>{prompt}</p>
          )}
        </div>
      </div>

      {/* Assistant Response */}
      <div className="assistant-bubble">
        <div className={`assistant-bubble-inner ${isDark ? 'dark' : 'light'}`}>
          <div className="bubble-header">
            <p style={{ color: isDark ? '#a78bfa' : '#6366f1' }}>TallyFormer</p>
            <div className="bubble-actions">
              <button
                onClick={() => copyText(response || '', setCopiedResponse)}
                disabled={isLoading}
                className={`action-btn ${isDark ? 'dark' : ''}`}
              >
                {copiedResponse ? <Check size={16} /> : <Copy size={16} />}
              </button>
              <button
                onClick={() => onRegenerate(index)}
                disabled={isLoading}
                className={`action-btn ${isDark ? 'dark' : ''}`}
              >
                <RotateCcw size={16} />
              </button>
            </div>
          </div>

          {isLoading ? (
             <div className="loading-icons">
              {[0, 1, 2].map((i) => (
                <img
                  key={i}
                  src={isDark ? darkIcon : lightIcon}
                  alt="TallyFormer Logo"
                  className="loading-icon"
                />
              ))}
            </div>
          ) : (
            <>
              <p style={{ whiteSpace: 'pre-wrap' }}>{response}</p>
              {metrics && (
                <div className={`metrics ${isDark ? 'dark' : 'light'}`}>
                    <span style={{ marginRight: '1rem' }}>Prompt: {metrics.prompt_tokens}</span>
                    <span style={{ marginRight: '1rem' }}>Generated: {metrics.generated_tokens}</span>
                    <span>Total Time : {metrics.total_time_sec.toFixed(2)} (s)</span>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;