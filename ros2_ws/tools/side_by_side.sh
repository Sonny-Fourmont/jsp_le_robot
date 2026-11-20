#!/usr/bin/env bash
# side_by_side.sh - puts the phone clip of the scene next to a recording made by record_demo /
# record_ouija (RViz + camera + detection), both scaled to 1080 px high. Sound of the phone kept.
#   side_by_side.sh telephone.mp4 ros2_ws/rec/ouija_reel.mp4 rendu.mp4 [decalage_s]
# decalage_s (optional): seconds to cut at the start of the phone clip so that both start together
# (positive) ; use a negative value to cut the start of the recording instead.
# Needs ffmpeg on the host, otherwise run it in the control container (ffmpeg is in the image):
#   docker compose exec control bash /ros2_ws/tools/side_by_side.sh /ros2_ws/rec/tel.mp4 /ros2_ws/rec/ouija_reel.mp4 /ros2_ws/rec/rendu.mp4
set -euo pipefail
phone=$1; rec=$2; out=$3; off=${4:-0}
cut_phone=0; cut_rec=0
if awk "BEGIN{exit !($off >= 0)}"; then cut_phone=$off; else cut_rec=${off#-}; fi
ffmpeg -y -loglevel error -stats \
    -ss "$cut_phone" -i "$phone" -ss "$cut_rec" -i "$rec" \
    -filter_complex "[0:v]scale=-2:1080,setsar=1[p];[1:v]scale=-2:1080,setsar=1[r];[p][r]hstack=inputs=2[v]" \
    -map "[v]" -map "0:a?" -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -c:a aac -shortest -movflags +faststart "$out"
echo "wrote $out"
