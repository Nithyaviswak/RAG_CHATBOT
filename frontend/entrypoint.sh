#!/bin/sh
set -eu

API_URL="${API_URL:-https://rag-chatbot-53e0.onrender.com/}"
ESCAPED_API_URL="$(printf '%s' "$API_URL" | sed 's/[&|]/\\&/g')"

sed -i "s|__API_URL__|${ESCAPED_API_URL}|g" /usr/share/nginx/html/index.html

exec "$@"
