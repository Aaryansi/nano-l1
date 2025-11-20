FROM golang:1.22-alpine AS build
WORKDIR /app

COPY go.mod ./
RUN go mod download

COPY . .
RUN go build -o engine ./cmd/engine

FROM alpine:3.20
WORKDIR /app
COPY --from=build /app/engine /app/engine

EXPOSE 8080
CMD ["/app/engine"]
