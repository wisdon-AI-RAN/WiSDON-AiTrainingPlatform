# Remove existing container and image if they exist
sudo docker stop aitrplat_flowerclient
sudo docker rm aitrplat_flowerclient
echo "Remove AiTrPlat_FlowerClient container finished !"
sudo docker rmi aitrplat_flowerclient:latest
echo "Remove AiTrPlat_FlowerClient image finished !"

# Remove existing container and image if they exist
sudo docker stop aitrplat_flowerserver
sudo docker rm aitrplat_flowerserver
echo "Remove AiTrPlat_FlowerServer container finished !"
sudo docker rmi aitrplat_flowerserver:latest
echo "Remove AiTrPlat_FlowerServer image finished !"