FROM node:20-alpine
WORKDIR /app

COPY package.json vite.config.js ./
RUN npm install

COPY index.html ./
COPY src ./src

EXPOSE 5173
CMD ["npm", "run", "dev"]
