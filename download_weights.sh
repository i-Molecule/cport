# Replace with your actual public Yandex.Disk link
PUBLIC_LINK="https://disk.yandex.ru/d/XrKLM7o_G_emww"

# Get direct download link
DOWNLOAD_URL=$(curl -sG "https://cloud-api.yandex.net/v1/disk/public/resources/download" \
  --data-urlencode "public_key=$PUBLIC_LINK" | jq -r '.href')

# Download the file
curl -L -o downloaded_file.zip "$DOWNLOAD_URL"

unzip downloaded_file.zip -d weights