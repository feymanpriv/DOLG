#!/bin/bash

input=$1
output=$2
for file in `ls $input` 
do  
    tar -xzf $input/$file -C $output
    echo "$file finished"
done

echo "over"
