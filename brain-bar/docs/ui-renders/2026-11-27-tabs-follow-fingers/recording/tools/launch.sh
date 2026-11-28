#!/bin/bash
# launch.sh before|after : quit any capture build, start this one on Overview with the calm fixture, place it at (612,33)
set -e
cd "$(dirname "$0")"
pkill -f 'bbui-phase18/app-.*/BrainBarCapture.app/Contents/MacOS/BrainBar' || true
sleep 1
case $1 in before) APP=../bbui-phase18/app-before-7ee776fb/BrainBarCapture.app;; after) APP=../bbui-phase18/app-after-388b358b/BrainBarCapture.app;; esac
BRAINBAR_DESIGN_FIXTURE=calm BRAINBAR_DESIGN_OPEN=window BRAINBAR_DESIGN_TAB=overview BRAINBAR_DESIGN_NOW=2026-11-27T21:00:00Z \
  "$APP/Contents/MacOS/BrainBar" > "app-$1.log" 2>&1 &
sleep 4
./drv place 612 33; sleep 0.5; ./drv bounds
