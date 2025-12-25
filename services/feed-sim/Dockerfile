FROM golang:1.23-alpine AS build
WORKDIR /app

# Copy BOTH go.mod and go.sum
COPY go.mod go.sum ./
RUN go mod download

COPY cmd ./cmd
COPY internal ./internal

RUN go build -o replay ./cmd/replay

FROM alpine:3.20
WORKDIR /app
COPY --from=build /app/replay /app/replay
CMD ["/app/replay"]
