#!/usr/bin/bash

pts=(128 512 2048)
dim=(8 32 128)
clr=(4 16 64)

echo "n_pts,n_dims,n_clrs,time"

for p in ${pts[@]}; do
    for d in ${dim[@]}; do
        for c in ${clr[@]}; do
            # echo "---------- ./$1 $p $d $c ----------"
            ./$1 $p $d $c
        done
    done
done
