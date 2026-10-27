#!/bin/zsh
# usage: adsb_worker.sh <worker_index> <n_workers>
cd /private/tmp/claude-501/-Users-jcool-Desktop/eceabdd0-f3e3-4f87-9eea-d15c83637308/scratchpad
i=0
while read d; do
  if [ $((i % $2)) -eq $1 ]; then
    if [ ! -f data/raw/adsb_traces/day_$d/.done ]; then ./code/adsb_pull.sh $d && touch data/raw/adsb_traces/day_$d/.done; fi
  fi
  i=$((i+1))
done < data/adsb_queue.txt
