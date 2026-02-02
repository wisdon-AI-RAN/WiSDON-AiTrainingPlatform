# Run the Mongo DB container
docker run --network aitrplat \
    --name mongodb \
    --detach \
    -p 27017:27017 \
    -v mongodb-data:/data/db \
    mongo
echo "Run MongoDB container finished !"

# Run the Computation Platform Common DB Mongo DB container
docker run --name mongodb_compcommondb \
    --detach \
    -p 27019:27017 \
    -v mongodb-compcommondb-data:/data/db \
    mongo
echo "Run Computation Platform Common DB MongoDB container finished !"

# Run the AI Training Platform Common DB Mongo DB container
docker run --name mongodb_aitrplatcommondb \
    --detach \
    -p 27018:27017 \
    -v mongodb-aitrplatcommondb-data:/data/db \
    mongo
echo "Run AI Training Platform Common DB MongoDB container finished !"