#!/usr/bin/bash

pts=(1024 4096 16384)
dim=(128 512 2048)
clr=(64 256 1024)

echo "n_pts,n_dims,n_clrs,time"

for p in ${pts[@]}; do
    for d in ${dim[@]}; do
        for c in ${clr[@]}; do
            # echo "---------- ./$1 $p $d $c ----------"
            ./$1 $p $d $c
        done
    done
done

