#!/bin/bash
# record.sh <out-prefix> <TABS> : SCK recording (120 Hz) of the window's right part (window x 280..900 pt) plus the
# reference strip below it, while drv runs all 8 gestures
cd "$(dirname "$0")"
./ref 900 780 600 37 &
sleep 0.5
./sck "$1.mov" 33 892 33 620 764 2> "$1.rec.log" &
sleep 1.5
TABS="$2" TABY=26 ./drv run "$1.events"
wait
cat "$1.rec.log"
