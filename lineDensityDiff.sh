#!/bin/bash
module load gdal2-stack
LineDensity=LineDensity
current_dir=$(pwd)
script_dir=$(dirname $0)
if [ "$#" -eq 8 ]; then
 LineDataset1=$1
 LineDataset2=$2
 cellsize=$3
 radius=$4
 xmin=$5
 ymin=$6
 xmax=$7
 ymax=$8
 outdir=$(dirname $(dirname ${LineDataset1}))
echo outdir is ${outdir}

outdir=${outdir}/results
if [ ! -d $outdir ]; then
 mkdir $outdir
#else
# rm -r $outdir
# mkdir $outdir
fi
 LineDensity1=${outdir}/$(basename ${LineDataset1} .shp)_density.tif
 LineDensity2=${outdir}/$(basename ${LineDataset2} .shp)_density.tif
 
 diff_density=${outdir}/diff_density.tif
 echo Output Directory $outdir
 echo Compare $LineDataset1 and $LineDataset2 with cellsize=$cellsize and radius=$radius 
 echo Bounding Box "<$xmin,$ymin,$xmax,$ymax>"
 echo Calculating line density for $LineDataset1 in meters per square meter
 $LineDensity $LineDataset1 $LineDensity1 $cellsize $radius $xmin $ymin $xmax $ymax
 echo Calculating line density for $LineDataset2 in meters per square meter
 $LineDensity $LineDataset2 $LineDensity2 $cellsize $radius $xmin $ymin $xmax $ymax
 LineDense1000=${outdir}/$(basename ${LineDataset1} .shp)1000_density.tif
 python gdal_calc.py -A $LineDensity1 --outfile=${LineDense1000} --calc="A*1000.000"
 rm ${LineDensity1}
 mv ${LineDense1000} ${LineDensity1}
 LineDense1000_1=${outdir}/$(basename ${LineDataset2} .shp)1000_density.tif
 python gdal_calc.py -A $LineDensity2 --outfile=${LineDense1000_1} --calc="A*1000.000"
 rm ${LineDensity2}
 mv ${LineDense1000_1} ${LineDensity2}
# LVS moving the differencing to the CLC computation because a 4x4 focal mean smoothing is needed for line density datasets.
# echo Calculating difference of line density
# python gdal_calc.py -A $LineDensity1 -B $LineDensity2 --outfile=${diff_density} --calc="(A-B)*1000.000"
else
 echo "wrong parameter"
fi

