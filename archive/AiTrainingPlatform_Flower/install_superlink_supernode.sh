# Run the SuperLink container
docker run --rm \
      -p 9091:9091 -p 9092:9092 -p 9093:9093 \
      --network aitrplat \
      --name superlink \
      --detach \
      flwr/superlink:1.25.0 \
      --insecure \
      --isolation \
      process
echo "Run SuperLink container finished !"

# Build and run the SuperNode container
docker run --rm \
    -p 9094:9094 \
    --network aitrplat \
    --name supernode-1 \
    --detach \
    flwr/supernode:1.25.0  \
    --insecure \
    --superlink superlink:9092 \
    --node-config "partition-id=0 num-partitions=1" \
    --clientappio-api-address 0.0.0.0:9094 \
    --isolation process
echo "Run SuperNode container finished !"

# docker run --rm \
#     -p 9095:9095 \
#     --network flwr-network \
#     --name supernode-2 \
#     --detach \
#     flwr/supernode:1.25.0  \
#     --insecure \
#     --superlink superlink:9092 \
#     --node-config "partition-id=1 num-partitions=2" \
#     --clientappio-api-address 0.0.0.0:9095 \
#     --isolation process
# echo "Run SuperNode container finished !"