#!/bin/bash
module purge
module load gdal2-stack
module load Python/2.7.9-goolf-1.7.20
module load LineDensity2

unset PYTHONPATH
source /projects/lstan/uiuc-clc/env/bin/activate

h=$1
nhd=NHDFLowline_071200011703.shp
eds=3m_DEM_frmLidr_channelNetwork_10ksk_clip.shp
wps=wps07120001_w_aps_and_eds_cl.shp
bnd=dem_boundary.shp

cwd=$(dirname $0)
if [ -z "$1" ]
then 
    echo no input
    exit
fi
datafolder=$cwd/$1
if [ ! -d "$datafolder" ]
then
    echo No folder named "$datafolder"
    exit
fi
echo 'Working on data in folder '$datafolder
outdir=${datafolder}/results
if [ ! -d $outdir ]; then
 mkdir $outdir
else
 rm -r $outdir
 mkdir $outdir
fi
script_dir=$(dirname $cwd)
cellsize=18
radius=18
#xmin=260350
#ymin=4472061
#xmax=385210
#ymax=4527973
#${script_dir}/lineDensityDiff.sh $shp1 $shp2 $cellsize $radius $xmin $ymin $xmax $ymax

shp1=$datafolder/streams/$nhd
echo Shape1 is $shp1
shp2=$datafolder/streams/$eds
echo Shape2 is $shp2
shp3=$datafolder/streams/$wps
echo Shape3 is $shp3
shp4=$datafolder/streams/$bnd
echo Shape4 is $shp4
if [ ! -e $shp1 ]
then
    echo $shp1 does not exist
    exit
fi
if [ ! -e $shp2 ]
then
    echo $shp2 does not exist
    exit
fi
if [ ! -e $shp3 ]
then
    echo $shp3 does not exist
    exit
fi
if [ ! -e $shp4 ]
then
    echo $shp4 does not exist
    exit
fi
echo script_dir is $script_dir
#python $script_dir/linedensity.py $shp1 $shp2 $cellsize $radius
#${script_dir}/lineDensityDiff.sh $shp1 $shp2 $cellsize $radius $xmin $ymin $xmax $ymax
python -W ignore $script_dir/CLC_w_waterpolys_lvs_mp_Panther_rect.py $shp1 $shp2 $shp3 $shp4 $cellsize $radius
