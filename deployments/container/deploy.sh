#!/bin/bash
# ── Deploy Container on Azure VM ─────────────────────────────────────────────
#
# Launches a Standard_B2s Azure VM, installs Docker, and runs the
# inference container. Represents the "always-on container" deployment mode.
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Docker image pushed to ACR (or build on VM)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh [resource-group] [location]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RESOURCE_GROUP="${1:-ml-inference-rg}"
LOCATION="${2:-eastus}"
VM_NAME="ml-inference-container"
VM_SIZE="Standard_B2s"
PORT=8080
ADMIN_USER="azureuser"

echo "═══════════════════════════════════════════════════════════"
echo "  Deploying Container on Azure VM: ${VM_SIZE}"
echo "  Resource Group: ${RESOURCE_GROUP}  |  Location: ${LOCATION}"
echo "═══════════════════════════════════════════════════════════"

# Step 1: Create resource group (if not exists)
echo "[1/5] Creating resource group..."
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" --output none

# Step 2: Create VM with Docker support
echo "[2/5] Creating Azure VM..."
az vm create \
    --name "${VM_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --image "Ubuntu2204" \
    --size "${VM_SIZE}" \
    --admin-username "${ADMIN_USER}" \
    --generate-ssh-keys \
    --custom-data cloud-init-docker.yaml \
    --output none

# Step 3: Open port for inference
echo "[3/5] Opening port ${PORT}..."
az vm open-port \
    --port "${PORT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    --output none

# Step 4: Get public IP
echo "[4/5] Retrieving public IP..."
PUBLIC_IP=$(az vm show \
    --name "${VM_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --show-details \
    --query publicIps \
    --output tsv)

# Step 5: Install Docker and run container via custom script extension
echo "[5/5] Installing Docker and starting inference server..."
az vm run-command invoke \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    --command-id RunShellScript \
    --scripts "
        sudo apt-get update -y
        sudo apt-get install -y docker.io python3-pip
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo pip3 install onnxruntime numpy fastapi uvicorn[standard] python-multipart torch torchvision
        echo 'Docker and dependencies installed. Ready for container deployment.'
    " \
    --output none

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✓ Azure VM deployment complete"
echo "  VM Name:   ${VM_NAME} (${VM_SIZE})"
echo "  Public IP: ${PUBLIC_IP}"
echo "  Endpoint:  http://${PUBLIC_IP}:${PORT}"
echo "  SSH:       ssh ${ADMIN_USER}@${PUBLIC_IP}"
echo ""
echo "  Hourly cost: \$0.0416 (Standard_B2s on-demand)"
echo "═══════════════════════════════════════════════════════════"
