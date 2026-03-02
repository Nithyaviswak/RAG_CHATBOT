FROM nginx:1.27-alpine

WORKDIR /usr/share/nginx/html

COPY index.html .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
