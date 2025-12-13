
import React, { useState, useEffect, useRef, useContext } from 'react';
import { Link } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import MessageBubble from './MessageBubble';
import SettingsModal from './SettingsModal';
import TokenCounter from './TokenCounter';
import { showError } from './ErrorToast';
import { Settings, Sun, Moon, Send, Loader } from 'lucide-react';
import { ThemeContext } from '../App';
import darkIcon from '../assets/icon-dark-514-514.jpg';
import lightIcon from '../assets/icon-light-514-514.jpg';

const DEFAULT_SETTINGS = {
  model_name: 'sft',
  max_new_tokens: 20,
  temperature: 0.9,
  topk: 500,
  topp: 0.9,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  mode: 'combined',
  return_metrics: true
};

const ChatPage = () => {
  const { theme, toggleTheme } = useContext(ThemeContext);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [totalTokens, setTotalTokens] = useState(0);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  const isDark = theme === 'dark';

  useEffect(() => {
    const savedSettings = localStorage.getItem('tallyformer_settings');
    const savedMessages = localStorage.getItem('tallyformer_messages');
    const savedTokens = localStorage.getItem('tallyformer_tokens');
    if (savedSettings) setSettings(JSON.parse(savedSettings));
    if (savedMessages) setMessages(JSON.parse(savedMessages));
    if (savedTokens) setTotalTokens(parseInt(savedTokens));
  }, []);

  useEffect(() => {
    localStorage.setItem('tallyformer_settings', JSON.stringify(settings));
    localStorage.setItem('tallyformer_messages', JSON.stringify(messages));
    localStorage.setItem('tallyformer_tokens', totalTokens.toString());
  }, [settings, messages, totalTokens]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const generateResponse = async (prompt, targetIndex) => {
    if (isGenerating) return;
    setIsGenerating(true);
    if (abortControllerRef.current) abortControllerRef.current.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
     
    const payload = { prompt, ...settings };
    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

      const res = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      if (!res.ok) throw new Error((await res.json()).detail || 'Server error');
      const data = await res.json();
      const metrics = data.metrics || null;
      const newTokens = metrics ? metrics.prompt_tokens + metrics.generated_tokens : 0;
      setTotalTokens(prev => prev + newTokens);
      setMessages(prev => {
        let trimmedResponse = data.generated_text.trimStart();
        if (trimmedResponse.length && ['.', '?', '!',','].includes(trimmedResponse[0])) {
          trimmedResponse = trimmedResponse.slice(1).trimStart();
        }
        const updated = [...prev];
        updated[targetIndex] = { prompt, response: trimmedResponse, metrics, isLoading: false };
        return updated;
      });
    } catch (err) {
      if (err.name !== 'AbortError') showError(err.message || 'Generation failed.');
      setMessages(prev => prev.slice(0, targetIndex));
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const startGeneration = (prompt, index) => {
    setMessages(prev => [...prev.slice(0, index), { prompt, response: '', isLoading: true }]);
    generateResponse(prompt, index);
  };

  const handleSend = () => {
    if (!input.trim() || isGenerating) return;
    const prompt = input.trim();
    setInput('');
    startGeneration(prompt, messages.length);
  };

  const handleRegenerate = (index) => startGeneration(messages[index].prompt, index);
  const handleEditSave = (index, newPrompt) => {
    if (newPrompt === messages[index].prompt) return;
    startGeneration(newPrompt, index);
  };

  return (
    <div className="container">
      <Toaster position="top-center" />
      {/* Header */}
      <header className={`header ${isDark ? 'dark' : 'light'}`}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <Link to="/welcome" className="back-link">← Back</Link>
        </div>

        <div className="header-logo" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <h1>TallyFormer</h1>
        </div>
        <img 
          src={isDark ? darkIcon : lightIcon} 
          alt="TallyFormer Logo" 
          className="app-logo" 
          style={{ width: '50px', height: '50px' }} 
        />
        <div>
          <button onClick={toggleTheme} className={`theme-btn ${isDark ? 'dark' : 'light'}`}>
            {isDark ? <Sun size={22} className="icon-light"/> : <Moon size={22} className="icon-light"/>}
          </button>
        </div>
      </header>

      <TokenCounter totalTokens={totalTokens} settings={settings} />
      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Welcome to TallyFormer</h2>
            <p>Start a conversation below</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} index={i} onRegenerate={handleRegenerate} onEditSave={handleEditSave} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div className={`input-bar ${isDark ? 'dark' : 'light'}`}>
        <div className={`input-container ${isDark ? 'dark' : 'light'}`}>
          <div className="controls-row">
            <select
              value={settings.model_name}
              onChange={(e) => setSettings(prev => ({ ...prev, model_name: e.target.value }))}
              disabled={isGenerating}
              className={`select-model ${isDark ? 'dark' : 'light'}`}
            >
              <option value="pretrain">Pretrain</option>
              <option value="distilled">Distilled</option>
              <option value="sft">SFT (Best)</option>
            </select>
            <button
              onClick={() => setShowSettings(true)}
              className={`settings-btn ${isDark ? 'dark' : 'light'}`}
            >
              <Settings size={20} className="icon-light" />
            </button>

          </div>

          <div className="input-area">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Type your message..."
              className={`textarea-input ${isDark ? 'dark' : 'light'}`}
              rows="1"
              style={{ minHeight: '50px' }}
              disabled={isGenerating}
            />
            <button
              onClick={handleSend}
              disabled={isGenerating || !input.trim()}
              className="send-btn"
            >
              {isGenerating ? <Loader className="animate-spin w-5 h-5" /> : <Send className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      <SettingsModal isOpen={showSettings} onClose={() => setShowSettings(false)} settings={settings} onUpdate={setSettings} />
    </div>
  );
};

export default ChatPage;