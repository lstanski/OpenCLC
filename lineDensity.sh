#!/bin/bash
module purge
module load gdal2-stack
module load Python/2.7.9-goolf-1.7.20
module load LineDensity2
LineDensity=LineDensity

unset PYTHONPATH
source /projects/lstan/uiuc-clc/env/bin/activate

current_dir=$(pwd)
script_dir=$(dirname $0)

if [ "$#" -eq 4 ]; then
 LineDataset=$1
 Output=$2
 cellsize=$3
 radius=$4
 echo "using $LineDensity"
 echo "input=$LineDataset output=$Output cellsize=$cellsize radius=$radius"
 $LineDensity $LineDataset $Output $cellsize $radius
elif [ "$#" -eq 8 ]; then
 LineDataset=$1
 Output=$2
 cellsize=$3
 radius=$4
 xmin=$5
 ymin=$6
 xmax=$7
 ymax=$8
 echo "using $LineDensity"
 echo "input=$LineDataset output=$Output cellsize=$cellsize radius=$radius bounding box: <$xmin, $ymin, $xmax, $ymax>"
 $LineDensity $LineDataset $Output $cellsize $radius $xmin $ymin $xmax $ymax
else
 echo "wrong parameter"
fi

