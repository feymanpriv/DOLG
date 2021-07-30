#!/bin/bash

cat file.txt | while read line
do  
    save_path="./" 
    #mkdir -p $save_path
    src="http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/$line"
    wget $src $save_path 
    echo "done!"

done

echo "over"
