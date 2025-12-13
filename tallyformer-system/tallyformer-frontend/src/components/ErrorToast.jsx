import React from 'react';
import toast from 'react-hot-toast';

export const showError = (message) => {
  toast.error(message, {
    duration: 6000,
    position: 'top-center',
    style: {
      background: '#1f1f1f',
      color: '#fff',
      border: '1px solid #ef4444',
      borderRadius: '12px',
      padding: '16px',
      maxWidth: '500px',
    },
  });
};

export const showSuccess = (message) => {
  toast.success(message, {
    duration: 4000,
    position: 'top-center',
    style: {
      background: '#1f1f1f',
      color: '#fff',
      border: '1px solid #10b981',
      borderRadius: '12px',
    },
  });
};