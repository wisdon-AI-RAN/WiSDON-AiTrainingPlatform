#!/bin/bash
# Start SuperNode for single-node Flower deployment

docker run --rm \
    --network aitrplat \
    --name supernode-1 \
    --detach \
    -p 9094:9094 \
    flwr/supernode:1.25.0 \
    --insecure \
    --superlink superlink:9092 \
    --clientappio-api-address 0.0.0.0:9094

echo "SuperNode started successfully!"
