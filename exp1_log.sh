# run this bash at exp1../trials/ to get average robustness across trials

awk '
/^pedestrian[[:space:]]/ {ped += $2; np++}
/^ambulance[[:space:]]/  {amb += $2; na++}
/^lane[[:space:]]/       {lane += $2; nl++}
END {
    printf "pedestrian average min robustness: %.4f  (%d trials)\n", ped/np, np
    printf "ambulance  average min robustness: %.4f  (%d trials)\n", amb/na, na
    printf "lane       average min robustness: %.4f  (%d trials)\n", lane/nl, nl
}' */robustness_summary.txt