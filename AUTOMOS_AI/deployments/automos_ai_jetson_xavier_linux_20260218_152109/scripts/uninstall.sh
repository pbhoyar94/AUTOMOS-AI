#!/bin/bash
# AUTOMOS AI Uninstallation Script

set -e

echo "Uninstalling AUTOMOS AI..."

# Stop service
systemctl stop automos_ai 2>/dev/null || true
systemctl disable automos_ai 2>/dev/null || true

# Remove files
rm -rf /opt/automos_ai
rm -rf /etc/automos_ai
rm -rf /var/log/automos_ai
rm -f /usr/local/bin/automos_ai
rm -f /etc/systemd/system/automos_ai.service

# Remove user
userdel automos 2>/dev/null || true

# Reload systemd
systemctl daemon-reload

echo "AUTOMOS AI uninstalled successfully!"
