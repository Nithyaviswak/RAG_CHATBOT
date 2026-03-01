
API_URL=${API_URL:-http://localhost:5000/api}
sed -i "s|window.API_BASE || window.ENV_API_URL || 'http://localhost:5000/api'|'${API_URL}'|g" /usr/share/nginx/html/index.html
exec "$@"