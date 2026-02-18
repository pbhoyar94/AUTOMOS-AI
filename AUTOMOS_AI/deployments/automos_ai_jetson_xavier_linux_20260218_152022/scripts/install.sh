#!/bin/bash
# AUTOMOS AI Installation Script
# Platform: jetson_xavier
# Target OS: linux

set -e

echo "Installing AUTOMOS AI for jetson_xavier (linux)"

# Check permissions
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

# Create directories
mkdir -p /opt/automos_ai
mkdir -p /var/log/automos_ai
mkdir -p /etc/automos_ai
mkdir -p /usr/local/bin

# Copy files
echo "Copying application files..."
cp -r bin/* /opt/automos_ai/bin/
cp -r lib/* /opt/automos_ai/lib/
cp -r config/* /etc/automos_ai/
cp -r models/* /opt/automos_ai/models/

# Set permissions
chmod +x /opt/automos_ai/bin/*
chmod -R 755 /opt/automos_ai/
chmod -R 755 /etc/automos_ai/

# Create user
if ! id "automos" &>/dev/null; then
    useradd -r -s /bin/false automos
fi
chown -R automos:automos /opt/automos_ai/
chown -R automos:automos /var/log/automos_ai/

# Install dependencies
echo "Installing dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y python3 python3-pip opencv-python
elif command -v yum &> /dev/null; then
    yum install -y python3 python3-pip opencv-python
fi

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
fi

# Install service
if [ -f "scripts/automos_ai.service" ]; then
    cp scripts/automos_ai.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable automos_ai
fi

# Create startup script
cat > /usr/local/bin/automos_ai << 'EOF'
#!/bin/bash
/opt/automos_ai/bin/automos_ai "$@"
EOF
chmod +x /usr/local/bin/automos_ai

echo "Installation completed successfully!"
echo "Start AUTOMOS AI with: systemctl start automos_ai"
echo "Check status with: systemctl status automos_ai"
