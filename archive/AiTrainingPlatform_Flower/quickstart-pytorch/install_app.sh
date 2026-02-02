# Build and run the SuperExec container to execute ServerApps 
docker stop $(docker ps -a -q --filter ancestor=flwr_superexec:0.0.1)
docker build -f superexec.Dockerfile -t flwr_superexec:0.0.1 .
docker run --rm \
    --network flwr-network \
    --name superexec-serverapp \
    --detach \
    flwr_superexec:0.0.1 \
    --insecure \
    --plugin-type serverapp \
    --appio-api-address superlink:9091
echo "Run SuperExec ServerApp container finished !"

# Build and run the SuperExec container to execute ClientApps 
# Use same image as serverapp
docker run --rm \
    --network flwr-network \
    --name superexec-clientapp-1 \
    --detach \
    flwr_superexec:0.0.1 \
    --insecure \
    --plugin-type clientapp \
    --appio-api-address supernode-1:9094
echo "Run SuperExec ClientApp container finished !"

# docker run --rm \
#     --network flwr-network \
#     --name superexec-clientapp-2 \
#     --detach \
#     flwr_superexec:0.0.1 \
#     --insecure \
#     --plugin-type clientapp \
#     --appio-api-address supernode-2:9095
# echo "Run SuperExec ClientApp container finished !"