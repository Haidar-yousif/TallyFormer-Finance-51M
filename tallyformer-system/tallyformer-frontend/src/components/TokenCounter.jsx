import React, { useContext } from 'react';
import { ThemeContext } from '../App';

const TokenCounter = ({ totalTokens, settings }) => {
  const { theme } = useContext(ThemeContext);
  const isDark = theme === 'dark';

  // hide completely if return_metrics is false
  if (!settings?.return_metrics) {
    return null;
  }

  return (
    <div className={`token-counter ${isDark ? 'dark' : 'light'}`}>
      <span>Session Tokens</span>
      <span>{totalTokens.toLocaleString()}</span>
    </div>
  );
};

export default TokenCounter;