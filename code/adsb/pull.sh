#!/bin/zsh
cd /private/tmp/claude-501/-Users-jcool-Desktop/eceabdd0-f3e3-4f87-9eea-d15c83637308/scratchpad/downloads/adsb
d=$1; y=${d:0:4}; out=day_$d; mkdir -p $out
split=""
for suf in 0 0tmp 1; do
  tag="v$d-planes-readsb-prod-$suf"; base="https://github.com/adsblol/globe_history_$y/releases/download/$tag/$tag.tar"
  code=$(curl -sIL -o /dev/null -w '%{http_code}' "$base.aa"); [ "$code" = "200" ] && { split=1; break; }
  code=$(curl -sIL -o /dev/null -w '%{http_code}' "$base"); [ "$code" = "200" ] && { split=0; break; }
  tag=""
done
[ -z "$tag" ] && { echo "$d NO_TAG" >> pull.log; exit 1; }
want=('traces/c6/trace_full_a034c6.json' 'traces/fe/trace_full_a037fe.json' 'traces/b5/trace_full_a03bb5.json' 'traces/6c/trace_full_a03f6c.json' 'traces/bb/trace_full_a04ebb.json' 'traces/82/trace_full_a05882.json' 'traces/38/trace_full_a0c538.json' 'traces/d1/trace_full_a168d1.json' 'traces/86/trace_full_a31e86.json' 'traces/05/trace_full_a34605.json' 'traces/cc/trace_full_a34fcc.json' 'traces/83/trace_full_a35383.json' 'traces/3a/trace_full_a3573a.json' 'traces/a8/trace_full_a35ea8.json' 'traces/5f/trace_full_a3625f.json' 'traces/16/trace_full_a36616.json' 'traces/94/trace_full_a37394.json' 'traces/4b/trace_full_a3774b.json' 'traces/02/trace_full_a37b02.json' 'traces/70/trace_full_a38270.json' 'traces/27/trace_full_a38627.json' 'traces/b0/trace_full_a38cb0.json' 'traces/95/trace_full_a38d95.json' 'traces/4c/trace_full_a3914c.json' 'traces/cf/trace_full_a3a5cf.json' 'traces/14/trace_full_a3ad14.json' 'traces/42/trace_full_a3e542.json' 'traces/80/trace_full_a40b80.json' 'traces/ff/trace_full_a432ff.json' 'traces/0e/trace_full_a4340e.json' 'traces/fd/trace_full_a481fd.json' 'traces/4a/trace_full_a4834a.json' 'traces/5d/trace_full_a4955d.json' 'traces/70/trace_full_a49570.json' 'traces/71/trace_full_a49671.json' 'traces/80/trace_full_a49680.json' 'traces/89/trace_full_a4e489.json' 'traces/40/trace_full_a4e840.json' 'traces/9b/trace_full_a51e9b.json' 'traces/c4/trace_full_a523c4.json' 'traces/7b/trace_full_a5277b.json' 'traces/a4/trace_full_a55ba4.json' 'traces/a2/trace_full_a626a2.json' 'traces/98/trace_full_a6e598.json' 'traces/29/trace_full_a71429.json' 'traces/31/trace_full_a79531.json' 'traces/f2/trace_full_a7edf2.json' 'traces/54/trace_full_a7f054.json' 'traces/70/trace_full_a7fb70.json' 'traces/27/trace_full_a7ff27.json' 'traces/de/trace_full_a809de.json' 'traces/71/trace_full_a81571.json' 'traces/f0/trace_full_a83cf0.json' 'traces/6f/trace_full_a8646f.json' 'traces/ee/trace_full_a88bee.json' 'traces/6b/trace_full_a9026b.json' 'traces/f2/trace_full_a9a9f2.json' 'traces/0b/trace_full_ab600b.json' 'traces/c2/trace_full_ab63c2.json' 'traces/9e/trace_full_ab729e.json' 'traces/55/trace_full_ab7655.json' 'traces/0c/trace_full_ab7a0c.json' 'traces/49/trace_full_abb749.json' 'traces/ac/trace_full_abc5ac.json' 'traces/11/trace_full_abdb11.json' 'traces/8b/trace_full_ac1b8b.json' 'traces/34/trace_full_ac2234.json' 'traces/59/trace_full_ac2d59.json' 'traces/97/trace_full_ac4597.json' 'traces/4e/trace_full_ac494e.json' 'traces/05/trace_full_ac4d05.json' 'traces/ef/trace_full_ac9bef.json' 'traces/5d/trace_full_aca35d.json' 'traces/dc/trace_full_aca5dc.json' 'traces/4f/trace_full_ad114f.json' 'traces/06/trace_full_ad1506.json' 'traces/e3/trace_full_ad8fe3.json' 'traces/24/trace_full_adb424.json' 'traces/02/trace_full_add702.json')
if [ "$split" = "1" ]; then
  ( curl -sL "$base.aa"; curl -sL "$base.ab" ) | tar -xf - -C $out $want 2>>tar_err.log
else
  curl -sL "$base" | tar -xf - -C $out $want 2>>tar_err.log
fi
echo "$d $tag split=$split $(find $out -name '*.json' | wc -l | tr -d ' ') traces $(du -sh $out | cut -f1)" >> pull.log
exit 0
