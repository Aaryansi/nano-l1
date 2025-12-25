#!/usr/bin/env bash
set -e

BROKER=${BROKER:-kafka:9092}

# bitnami kafka path
KAFKA_TOPICS="/opt/bitnami/kafka/bin/kafka-topics.sh"

echo "Creating topics on $BROKER ..."

$KAFKA_TOPICS --bootstrap-server "$BROKER" --create --if-not-exists \
  --topic ticks --partitions 1 --replication-factor 1

$KAFKA_TOPICS --bootstrap-server "$BROKER" --create --if-not-exists \
  --topic book_updates --partitions 1 --replication-factor 1

$KAFKA_TOPICS --bootstrap-server "$BROKER" --create --if-not-exists \
  --topic trades --partitions 1 --replication-factor 1

echo "Done."
