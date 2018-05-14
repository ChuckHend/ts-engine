# wakeup at 11pm CST
cd /mnt/raid0/projects/forecast-engine/src

python monitor.py

# standby until tomorrow at 11pm
sudo rtcwake -m standby -l -t $(date +%s -d 'tomorrow 23:00') && chronie.sh