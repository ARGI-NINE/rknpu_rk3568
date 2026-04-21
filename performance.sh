#!/usr/bin/env bash
set -e

echo "CPU available frequency:"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
echo userspace | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor >/dev/null
echo 1992000 | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed >/dev/null
echo "CPU current frequency:"
sudo cat /sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq

echo "NPU available frequency:"
sudo cat /sys/class/devfreq/fde40000.npu/available_frequencies
echo userspace | sudo tee /sys/class/devfreq/fde40000.npu/governor >/dev/null
echo 900000000 | sudo tee /sys/class/devfreq/fde40000.npu/userspace/set_freq >/dev/null
echo "NPU current frequency:"
sudo cat /sys/class/devfreq/fde40000.npu/cur_freq
