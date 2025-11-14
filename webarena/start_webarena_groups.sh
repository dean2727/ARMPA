#!/bin/bash
# Spin up 8 WebArena groups with incremented ports per group in the EC2 instance (enabling batches/parallelism in ARMPA project)
# Base config
IP="18.117.150.206"
HOSTNAME="ec2-18-117-150-206.us-east-2.compute.amazonaws.com"

# Base ports for group 1
SHOPPING_PORT=7770
ADMIN_PORT=7780
FORUM_PORT=9999
GITLAB_PORT=8023

# Start the shared kiwix container once
echo "Starting shared kiwix33 (Wikipedia mirror)..."
docker start kiwix33 2>/dev/null || docker run -d --name kiwix33 -p 8888:80 kiwix_final_image

for GROUP in {1..1}; do
  echo ""
  echo "============================="
  echo "🚀 Starting GROUP $GROUP"
  echo "============================="

  SHOPPING_NAME="shopping_g${GROUP}"
  ADMIN_NAME="shopping_admin_g${GROUP}"
  FORUM_NAME="forum_g${GROUP}"
  GITLAB_NAME="gitlab_g${GROUP}"

  # Stop + remove old containers if they exist
  docker stop $SHOPPING_NAME $ADMIN_NAME $FORUM_NAME $GITLAB_NAME 2>/dev/null || true
  docker rm   $SHOPPING_NAME $ADMIN_NAME $FORUM_NAME $GITLAB_NAME 2>/dev/null || true

  # Run containers with incremented ports
  docker run -d --name $SHOPPING_NAME -p ${SHOPPING_PORT}:80 shopping_final_0712
  docker run -d --name $ADMIN_NAME   -p ${ADMIN_PORT}:80   shopping_admin_final_0719
  docker run -d --name $FORUM_NAME   -p ${FORUM_PORT}:80   postmill-populated-exposed-withimg
  docker run -d --name $GITLAB_NAME  -p ${GITLAB_PORT}:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start

  # --- Magento configuration ---
  echo "Configuring Magento for group $GROUP..."

  docker exec $SHOPPING_NAME /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${IP}:${SHOPPING_PORT}"
  docker exec $SHOPPING_NAME mysql -u magentouser -pMyPassword magentodb -e \
    "UPDATE core_config_data SET value='http://${IP}:${SHOPPING_PORT}/' WHERE path = 'web/secure/base_url';"

  docker exec $ADMIN_NAME php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
  docker exec $ADMIN_NAME php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
  docker exec $SHOPPING_NAME /var/www/magento2/bin/magento cache:flush

  docker exec $ADMIN_NAME /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${IP}:${ADMIN_PORT}"
  docker exec $ADMIN_NAME mysql -u magentouser -pMyPassword magentodb -e \
    "UPDATE core_config_data SET value='http://${IP}:${ADMIN_PORT}/' WHERE path = 'web/secure/base_url';"
  docker exec $ADMIN_NAME /var/www/magento2/bin/magento cache:flush

  # --- GitLab configuration ---
  echo "Configuring GitLab for group $GROUP..."
  docker exec $GITLAB_NAME sed -i "s|^external_url.*|external_url 'http://${HOSTNAME}:${GITLAB_PORT}'|" /etc/gitlab/gitlab.rb
  docker exec $GITLAB_NAME gitlab-ctl reconfigure

  # Echo summary for the group
  echo "✅ GROUP $GROUP running:"
  echo "   Shopping:       http://${IP}:${SHOPPING_PORT}"
  echo "   Shopping Admin: http://${IP}:${ADMIN_PORT}"
  echo "   Forum:          http://${IP}:${FORUM_PORT}"
  echo "   GitLab:         http://${IP}:${GITLAB_PORT}"
  echo ""

  # Increment ports by 1 for next group
  SHOPPING_PORT=$((SHOPPING_PORT + 1))
  ADMIN_PORT=$((ADMIN_PORT + 1))
  FORUM_PORT=$((FORUM_PORT + 1))
  GITLAB_PORT=$((GITLAB_PORT + 1))
done

cd /home/ubuntu/openstreetmap-website/

docker compose start

echo ""
echo "🎉 All 8 groups started successfully!"
echo "Kiwix available at: http://${IP}:8888"
