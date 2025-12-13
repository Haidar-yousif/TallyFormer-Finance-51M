import React, { useState } from 'react';

const SettingsModal = ({ isOpen, onClose, settings, onUpdate }) => {
  const [localSettings, setLocalSettings] = useState(settings);
  const [errors, setErrors] = useState({});

  const validate = () => {
    const newErrors = {};
    if (localSettings.max_new_tokens < 1 || localSettings.max_new_tokens > 250)
      newErrors.max_new_tokens = 'Must be between 1 and 250';
    if (localSettings.temperature < 0.1 || localSettings.temperature > 0.99)
      newErrors.temperature = 'Must be between 0.1 and 0.99';
    if (localSettings.topk < 1 || localSettings.topk > 50250)
      newErrors.topk = 'Must be between 1 and 50,250';
    if (localSettings.topp < 0.3 || localSettings.topp > 0.99)
      newErrors.topp = 'Must be between 0.3 and 0.99';
    if (localSettings.frequency_penalty < 0)
      newErrors.frequency_penalty = 'Must be ≥ 0';
    if (localSettings.presence_penalty < 0)
      newErrors.presence_penalty = 'Must be ≥ 0';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (key, value) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
    setErrors(prev => ({ ...prev, [key]: null }));
  };

  const handleSave = () => {
    if (validate()) {
      onUpdate(localSettings);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2 className="modal-title">Generation Settings</h2>
        <div className="modal-form">
         
          <div>
            <label className="modal-label">Max New Tokens (1–250)</label>
            <input
              type="number"
              min="1"
              max="250"
              value={localSettings.max_new_tokens}
              onChange={(e) => handleChange('max_new_tokens', parseInt(e.target.value) || 1)}
              className="modal-input"
            />
            {errors.max_new_tokens && <p className="error-text">{errors.max_new_tokens}</p>}
          </div>

          <div>
            <label className="modal-label">Temperature (0.1–0.99)</label>
            <input
              type="number"
              step="0.01"
              min="0.1"
              max="0.99"
              value={localSettings.temperature}
              onChange={(e) => handleChange('temperature', parseFloat(e.target.value) || 0.1)}
              className="modal-input"
            />
            {errors.temperature && <p className="error-text">{errors.temperature}</p>}
          </div>

          <div>
            <label className="modal-label">Top-K (1–50250)</label>
            <input
              type="number"
              min="1"
              max="50250"
              value={localSettings.topk}
              onChange={(e) => handleChange('topk', parseInt(e.target.value) || 1)}
              className="modal-input"
            />
            {errors.topk && <p className="error-text">{errors.topk}</p>}
          </div>

          <div>
            <label className="modal-label">Top-P (0.3–0.99)</label>
            <input
              type="number"
              step="0.01"
              min="0.3"
              max="0.99"
              value={localSettings.topp}
              onChange={(e) => handleChange('topp', parseFloat(e.target.value) || 0.3)}
              className="modal-input"
            />
            {errors.topp && <p className="error-text">{errors.topp}</p>}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div>
              <label className="modal-label">Frequency Penalty (≥0)</label>
              <input
                type="number"
                step="0.1"
                min="0"
                value={localSettings.frequency_penalty}
                onChange={(e) => handleChange('frequency_penalty', parseFloat(e.target.value) || 0)}
                className="modal-input"
              />
              {errors.frequency_penalty && <p className="error-text">{errors.frequency_penalty}</p>}
            </div>
            <div>
              <label className="modal-label">Presence Penalty (≥0)</label>
              <input
                type="number"
                step="0.1"
                min="0"
                value={localSettings.presence_penalty}
                onChange={(e) => handleChange('presence_penalty', parseFloat(e.target.value) || 0)}
                className="modal-input"
              />
              {errors.presence_penalty && <p className="error-text">{errors.presence_penalty}</p>}
            </div>
          </div>

          <div>
            <label className="modal-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              Mode
              <div className="tooltip-wrapper">
                <span className="tooltip-trigger">?</span>
                <div className="tooltip-box">
                  <div style={{ fontWeight: '600', marginBottom: '0.5rem' }}>Mode Explanation</div>
                  <div style={{ marginBottom: '0.5rem' }}>
                    <span style={{ color: '#93c5fd', fontWeight: '600' }}>Combined:</span> Top-p is applied inside Top-k filtering. Safer, more focused.
                  </div>
                  <div>
                    <span style={{ color: '#86efac', fontWeight: '600' }}>Independent:</span> Top-p on full logits. Freer, more creative.
                  </div>
                </div>
              </div>
            </label>
            <select
              value={localSettings.mode}
              onChange={(e) => handleChange('mode', e.target.value)}
              className="modal-input"
              style={{ appearance: 'none', paddingRight: '2rem' }}
            >
              <option value="combined">Combined</option>
              <option value="independent">Independent</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center' }}>
            <input
              type="checkbox"
              id="metrics"
              checked={localSettings.return_metrics}
              onChange={(e) => handleChange('return_metrics', e.target.checked)}
              style={{ width: '1.25rem', height: '1.25rem', marginRight: '0.75rem' }}
            />
            <label htmlFor="metrics" style={{ color: '#d1d5db' }}>Show token metrics</label>
          </div>
        </div>

        <div className="modal-actions">
          <button onClick={onClose} className="modal-btn-cancel">Cancel</button>
          <button onClick={handleSave} className="modal-btn-save">Save Settings</button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;