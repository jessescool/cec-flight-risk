#!/bin/zsh
cd /private/tmp/claude-501/-Users-jcool-Desktop/eceabdd0-f3e3-4f87-9eea-d15c83637308/scratchpad
OUT=data/nbm_kcec_nbs_12z.txt; DONE=data/nbm_done.txt; touch $DONE
d=$(date -j -f %Y%m%d 20240430 +%s)
end=$(date -j -f %Y%m%d 20200518 +%s)
while [ $d -ge $end ]; do
  key=$(date -j -r $d +%Y%m%d)
  if ! grep -q "^$key" $DONE; then
    blk=$(curl -s --max-time 180 "https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.$key/12/text/blend_nbstx.t12z" | awk '/^ KCEC /{f=1} f&&/^ *$/{exit} f')
    if [ -n "$blk" ]; then printf "#### %s 12Z\n%s\n" "$key" "$blk" >> $OUT; echo "$key" >> $DONE; else echo "$key MISSING" >> $DONE; fi
  fi
  d=$((d-86400))
done
