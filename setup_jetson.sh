#!/bin/bash
mkdir -p /home/jetson/robosafe
cd /home/jetson/robosafe
git clone https://github.com/arjuncs002/robosafe-backend .
pip3 install --break-system-packages ultralytics opencv-python pyserial requests
sudo cp robosafe.service /etc/systemd/system/robosafe.service
sudo systemctl daemon-reload
sudo systemctl enable robosafe
sudo systemctl start robosafe