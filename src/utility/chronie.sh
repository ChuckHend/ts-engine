cd /mnt/raid0/projects/forecast-engine/src

python monitor.py

cd /mnt/raid0/Projects/forecast-engine/src/utility

sudo rtcwake -m mem -l -t $(date +%s -d 'tomorrow 20:00') && sh chronie.sh