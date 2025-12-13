#!/bin/sh
# Replace API URL in the built index.html
: "${VITE_API_URL:?Need to set VITE_API_URL}"
echo "Replacing API URL with $VITE_API_URL"
sed -i "s|http://localhost:8000|$VITE_API_URL|g" /usr/share/nginx/html/assets/index.html

# Start nginx
nginx -g 'daemon off;'
