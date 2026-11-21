#!/usr/bin/env bash
# usage: telemetry.sh OUT  -- logs time, host RAM used (MB), chip temp (C) every 5 s
H=$(ls -d /sys/class/tenstorrent/tenstorrent\!0/device/hwmon/hwmon*)
while sleep 5; do echo "$(date +%T) ram_mb=$(free -m | awk "/Mem/{print \$3}") temp_c=$(( $(cat $H/temp1_input) / 1000 )) power_w=$(( $(cat $H/power1_input) / 1000000 ))"; done >> "$1"
