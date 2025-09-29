#!/bin/bash
# scripts/collector-health.sh
# Health check script for data collector container

# Check if Python process is running
if pgrep -f "python" > /dev/null; then
    echo "Collector process is running"
    exit 0
else
    echo "Collector process not found"
    exit 1
fi