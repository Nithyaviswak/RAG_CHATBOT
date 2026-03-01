
API_URL=${API_URL:-https://rag-chatbot-53e0.onrender.com/}
sed -i "s|window.API_BASE || window.ENV_API_URL || 'https://rag-chatbot-53e0.onrender.com/'|'${API_URL}'|g" /usr/share/nginx/html/index.html
exec "$@"