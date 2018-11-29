'''
CLC.py: CLC workflow
Author: Ting Li <tingli3@illinois.edu>, Larry Stanislawski <lstan@usgs.gov>
Date: 08/01/2017

'''

import os, string, time
# import rasterio, rasterio.features
from osgeo import gdal
from osgeo import ogr
import glob
import sys
from osgeo import osr
import numpy
import math
import scipy
from shapely.geometry import shape, mapping
from fishnet import fishnet
from skimage.measure import label, regionprops
import fiona
import subprocess

import conf


def WriteLineToFile(file, linest1):
    print(linest1)
    report = open(file, 'a')
    report.write(linest1 + "\n")
    report.close()


codedir = os.path.dirname(os.path.realpath(__file__))
workdir = os.getcwd()

if len(sys.argv) < 7:
    print('usage: {} <shapefile1> <shapefile2> <shapefile3> <cellsize> <radius>'.format(sys.argv[0]))
    sys.exit(-1)

unclipped_nhdData = sys.argv[1]
unclipped_edsData = sys.argv[2]
unclipped_wpsData = sys.argv[3]
unclipped_bndData = sys.argv[4]
cellsize = sys.argv[5]
radius = sys.argv[6]

# logfile
nhdFilename = string.split(os.path.basename(unclipped_nhdData), ".")[0]
edsFilename = string.split(os.path.basename(unclipped_edsData), ".")[0]
wpsFilename = string.split(os.path.basename(unclipped_wpsData), ".")[0]
bndFilename = string.split(os.path.basename(unclipped_bndData), ".")[0]
outdir = os.path.dirname(os.path.dirname(unclipped_nhdData)) + "/results"
clipped_nhdData = outdir + "/" + nhdFilename + ".shp"
clipped_edsData = outdir + "/" + edsFilename + ".shp"
match_edsData = outdir + "/" + edsFilename + "_match.shp"
#subbasin = os.path.basename(os.path.dirname(os.path.dirname(unclipped_nhdData)))
#subbasin = subbasin[len(subbasin) - 8:]
water_polys = os.path.dirname(unclipped_nhdData) + "/" + wpsFilename + ".shp"

watpol_layer = wpsFilename
bnd_shp = os.path.dirname(unclipped_nhdData)+"/"+bndFilename+".shp"

print "boundary is "+bndFilename
# sys.exit()
nhdDensity = outdir + "/" + nhdFilename + "_density.tif"
nhdDensity_t = outdir + "/" + nhdFilename + "_density_t.tif"
edsDensity = outdir + "/" + edsFilename + "_density.tif"
edsDensity_t = outdir + "/" + edsFilename + "_density_t.tif"
buffered_basin = outdir + "/" + edsFilename + "_buffered_basin.shp"
txt_results = outdir + "/clc_" + nhdFilename + "_" + edsFilename + "_results.txt"
outfile = outdir + "/clc_lvs_results_" + nhdFilename + "_" + edsFilename + ".txt"
print("Results written to " + outfile)
linest = "input line data is " + clipped_nhdData + " and " + clipped_edsData
WriteLineToFile(outfile, linest)
#WriteLineToFile(outfile, "Processing subbasin " + subbasin)
linest = "Cell size is "+cellsize+", and radius is "+radius
WriteLineToFile(outfile,linest)
use_waterpolys = 0
if os.path.isfile(water_polys):
    linest = "Using water polygon shapefile " + water_polys
    WriteLineToFile(outfile, linest)
    use_waterpolys = 1
if os.path.isfile(bnd_shp):
    linest = "Using boundary polygon shapefile " + bnd_shp
    WriteLineToFile(outfile, linest)
WriteLineToFile(outfile, "Run start" + str(time.ctime()))
start_time = time.time()
#
# Need to clip the line density datasets to the subbasin boundary
#
# Buffer the subbasin
#subbasin = '07100007'
# bnd_shp should have only one polygon
with fiona.open(bnd_shp) as wbd:
    bnd = next(iter(wbd))
# with fiona.open(conf.WBD_HUC8) as wbd:
    #bnd = next(iter(filter(lambda f: f['properties']['HUC8'] == subbasin, wbd)))
    buffered_geom = shape(bnd['geometry'])
    buffered_geom = buffered_geom.buffer(int(cellsize))
    bnd['geometry'] = mapping(buffered_geom)

    with fiona.open(buffered_basin, 'w', **wbd.meta) as out:
        out.write(bnd)

# Clip line
subprocess.call(['ogr2ogr', '-clipsrc', buffered_basin, clipped_nhdData, unclipped_nhdData])
subprocess.call(['ogr2ogr', '-clipsrc', buffered_basin, clipped_edsData, unclipped_edsData])

#
# Crop the nhd and eds to the buffered basin polygon.
#

### calculate line density and get the difference
inDriver = ogr.GetDriverByName("ESRI Shapefile")

nhdDataSource = inDriver.Open(clipped_nhdData, 0)
nhdLayer = nhdDataSource.GetLayer(0)
nhdExtent = nhdLayer.GetExtent()

edsDataSource = inDriver.Open(clipped_edsData, 0)
edsLayer = edsDataSource.GetLayer(0)
edsExtent = edsLayer.GetExtent()
# extent (xmin, xmax, ymin, ymax)
xmin = min(nhdExtent[0], edsExtent[0]) - float(cellsize)
xmax = max(nhdExtent[1], edsExtent[1]) + float(cellsize)
ymin = min(nhdExtent[2], edsExtent[2]) - float(cellsize)
ymax = max(nhdExtent[3], edsExtent[3]) + float(cellsize)
print "xmin " + str(round(xmin, 12)) + " ymin " + str(round(ymin, 12)) + " xmax " + str(
    round(xmax, 12)) + " ymax " + str(round(ymax, 12))
os.system(
    '{0}/lineDensityDiff.sh {1} {2} {3} {4} {5} {6} {7} {8}'.format(codedir, clipped_nhdData, clipped_edsData, cellsize,
                                                                    radius,
                                                                    xmin, ymin, xmax, ymax))
#sys.exit() #after linedensity

# set values outside subbasin to -9999 for both datasets.
subprocess.call(
    ["gdalwarp", "-cutline", buffered_basin, "-crop_to_cutline", nhdDensity, "-of", "GTiff", "-dstnodata", "-9999",
     "-overwrite", nhdDensity_t])
os.system("gdalmanage delete " + nhdDensity)
os.system("gdalmanage rename " + nhdDensity_t + " " + nhdDensity)

subprocess.call(
    ["gdalwarp", "-cutline", buffered_basin, "-crop_to_cutline", edsDensity, "-of", "GTiff", "-dstnodata", "-9999",
     "-overwrite", edsDensity_t])
os.system("gdalmanage delete " + edsDensity)
os.system("gdalmanage rename " + edsDensity_t + " " + edsDensity)
#sys.exit() # after clip
#
# Let's try to do a 3x3 mean filter for each line density dataset and then compute the diff density.
#
print outdir + ", " + nhdDensity
nhdDensityRaster = gdal.Open('{0}'.format(nhdDensity))
edsDensityRaster = gdal.Open('{0}'.format(edsDensity))
nhdDensityBand = nhdDensityRaster.GetRasterBand(1)
edsDensityBand = edsDensityRaster.GetRasterBand(1)
ndv = -9999
nhdDensityArray = nhdDensityBand.ReadAsArray().astype(numpy.float)
nhdMaskedArray = numpy.ma.masked_where(nhdDensityArray == ndv, nhdDensityArray)
edsDensityArray = edsDensityBand.ReadAsArray().astype(numpy.float)
edsMaskedArray = numpy.ma.masked_where(edsDensityArray == ndv, edsDensityArray)
# smooth 3x3
kernel = numpy.ones((3, 3))
nhdResult = scipy.ndimage.convolve(nhdMaskedArray, weights=kernel) / kernel.size
edsResult = scipy.ndimage.convolve(edsMaskedArray, weights=kernel) / kernel.size
nhdMaskedArray = None
edsMaskedArray = None
# write smoothed density raster1
geotransform = nhdDensityRaster.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
cols = nhdDensityRaster.RasterXSize
rows = nhdDensityRaster.RasterYSize
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
outRasterSRS = osr.SpatialReference()
outRasterSRS.ImportFromWkt(nhdDensityRaster.GetProjectionRef())
smdenseRaster1 = '{0}/smdense_raster1.tif'.format(outdir)
driver = gdal.GetDriverByName('GTiff')
denseClass = driver.Create(smdenseRaster1, cols, rows, 1, gdal.GDT_Float32)  # need a float driver
denseClass.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
denseClassBand = denseClass.GetRasterBand(1)
denseClass.SetProjection(outRasterSRS.ExportToWkt())
denseClassBand.WriteArray(nhdResult)
denseClassBand.FlushCache()
# 5/8/2018 lvs re-added
denseClass = None
denseClassBand = None
#
nhdResult = None
# Set values outside boundary to -19998 for the first smoothed raster dataset.
subprocess.call(
    ["gdalwarp", "-cutline", buffered_basin, "-crop_to_cutline", smdenseRaster1, "-of", "GTiff", "-dstnodata", "-19998",
     "-overwrite", nhdDensity])
# write smoothed density raster2
geotransform = edsDensityRaster.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
cols = edsDensityRaster.RasterXSize
rows = edsDensityRaster.RasterYSize
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
outRasterSRS = osr.SpatialReference()
outRasterSRS.ImportFromWkt(edsDensityRaster.GetProjectionRef())
smdenseRaster2 = '{0}/smdense_raster2.tif'.format(outdir)
driver = gdal.GetDriverByName('GTiff')
denseClass = driver.Create(smdenseRaster2, cols, rows, 1, gdal.GDT_Float32)  # need a float driver
denseClass.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
denseClassBand = denseClass.GetRasterBand(1)
denseClass.SetProjection(outRasterSRS.ExportToWkt())
denseClassBand.WriteArray(edsResult)
denseClassBand.FlushCache()
denseClass = None
denseClassBand = None
edsResult = None
# Set values outside boundary to -9999 for the second smoothed raster dataset.
subprocess.call(
    ["gdalwarp", "-cutline", buffered_basin, "-crop_to_cutline", smdenseRaster2, "-of", "GTiff", "-dstnodata", "-9999",
     "-overwrite", edsDensity])
#sys.exit() #after smooth and crop
#
# Compute difference raster (first smoothed dataset minus second smoothed dataset)
# Values outside the boundary should have -9999.
#
weight_tif = outdir + '/diff_density.tif'
temp_diff = outdir + '/temp_diff_density.tif'
# 02/06/2018 begin
# we need to get these 'keep = 1' polygons assigned 2 values in the diffClassArray (in Reclassify section).
# Let's try gdal_raster to create a new tif file with these these polygons having a values of 1.
watpol_tif = outdir + '/waterpolys.tif'
watpol_tif2 = outdir + '/waterpolys2.tif'
subprocess.call(
    ["gdal_rasterize", "-tr", str(pixelWidth), str(pixelHeight), "-burn", "2", "-where", "keep=1", "-a_nodata", "-9999",
     "-l", watpol_layer, water_polys, watpol_tif2])
# Set values outside boundary to -9999 for watpol_tif file.
subprocess.call(
    ["gdalwarp", "-cutline", buffered_basin, "-crop_to_cutline", watpol_tif2, "-of", "GTiff", "-dstnodata", "-9999",
     "-overwrite", watpol_tif])
# 02/06/2018 end
# subprocess.call(gdal_calc.py -A smdenseRaster1 -B smdenseRaster2 --outfile=weight_tif --calc="(A-B)"
os.system("python gdal_calc.py -A " + nhdDensity + " -B " + edsDensity + " --outfile=" + temp_diff + ' --calc="(A-B)"')
# Set values outside boundary to -9999 for diff density dataset.
subprocess.call(
    ["gdalwarp", "-cutline", buffered_basin, "-crop_to_cutline", temp_diff, "-of", "GTiff", "-dstnodata", "-9999",
     "-overwrite", weight_tif])
print(weight_tif + "," + workdir)
# weights = rasterio.open(glob.glob(weight_tif)[0])
# weight_ras = weights.read(masked=True)[0]
# ras_mean = weight_ras.mean()
# ras_std = weight_ras.std()
# ras_min = weight_ras.min()
# ras_max = weight_ras.max()
# weights.close()

diffRaster = gdal.Open('{0}/diff_density.tif'.format(outdir))
diffBand = diffRaster.GetRasterBand(1)
geotransform = diffRaster.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
cols = diffRaster.RasterXSize
rows = diffRaster.RasterYSize
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
outRasterSRS = osr.SpatialReference()
outRasterSRS.ImportFromWkt(diffRaster.GetProjectionRef())
diffArray = diffBand.ReadAsArray().astype(numpy.float)
maskeddiffArray = numpy.ma.masked_where(diffArray == ndv, diffArray)
diffRaster = None
### reclassify the difference raster ###
diffClassRaster = '{0}/diff_class.tif'.format(outdir)
driver = gdal.GetDriverByName('GTiff')
diffClass = driver.Create(diffClassRaster, cols, rows, 1, gdal.GDT_Int16)
diffClass.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
diffClassBand = diffClass.GetRasterBand(1)
diffClass.SetProjection(outRasterSRS.ExportToWkt())

sqDiff = numpy.square(maskeddiffArray)
ras_max = numpy.amax(maskeddiffArray)
ras_min = numpy.amin(maskeddiffArray)
ras_mean = numpy.mean(maskeddiffArray)
ras_std = numpy.std(maskeddiffArray)
ras_abs = numpy.absolute(maskeddiffArray)
ras_mean_avd = numpy.mean(ras_abs)
mean = numpy.mean(sqDiff)
RMSE = math.sqrt(mean)
NSSDA = 1.96 * RMSE
sqDiff = None
ras_abs = None
print ('NSSDA:' + str(NSSDA))

linest = "Minimum, Maximum, Mean, STD, RMSE, 95% NSSDA estimate, Mean Abs. Val. Diffs."
WriteLineToFile(outfile, linest)
linest = str(round(ras_min, 6)) + "," + str(round(ras_max, 6)) + "," + str(round(ras_mean, 6)) + "," + str(
    round(ras_std, 6))
linest = linest + "," + str(round(RMSE, 6)) + "," + str(round(NSSDA, 6)) + "," + str(round(ras_mean_avd, 6))
WriteLineToFile(outfile, linest)
# sys.exit()
print ("Reclassify...")
diffClassArray = numpy.full((rows, cols), 2)
diffClassArray[maskeddiffArray <= -NSSDA] = 1
diffClassArray[maskeddiffArray >= NSSDA] = 3
diffClassArray[(maskeddiffArray > -NSSDA) & (maskeddiffArray < NSSDA)] = 2

nhdLineDensity = os.path.splitext(os.path.basename(clipped_nhdData))[0] + '_density.tif'  # nodata = -19998
edsLineDensity = os.path.splitext(os.path.basename(clipped_edsData))[0] + '_density.tif'  # nodata = -9999

nhdDensityRaster = gdal.Open('{0}/{1}'.format(outdir, nhdLineDensity))
edsDensityRaster = gdal.Open('{0}/{1}'.format(outdir, edsLineDensity))
nhdDensityBand = nhdDensityRaster.GetRasterBand(1)
edsDensityBand = edsDensityRaster.GetRasterBand(1)
nhdDensityArray = nhdDensityBand.ReadAsArray().astype(numpy.float)
maskeddiffClassArray = numpy.ma.masked_where(nhdDensityArray == -19998, diffClassArray)
diffClassArray = maskeddiffClassArray
maskeddiffClassArray = None
maskeddensityArray1 = numpy.ma.masked_where(nhdDensityArray == -19998, nhdDensityArray)
edsDensityArray = edsDensityBand.ReadAsArray().astype(numpy.float)
maskeddensityArray2 = numpy.ma.masked_where(edsDensityArray == -9999, edsDensityArray)
nhdDensityRaster = None
edsDensityRaster = None
nhdDensityArray = None
edsDensityArray = None

# LVS 09/19/2017
# diffClassArray[densityArray1 < 0.000000001]=4
# diffClassArray[densityArray2 < 0.000000001]=4
diffClassArray[maskeddensityArray1 < 0.001] = 4
diffClassArray[maskeddensityArray2 < 0.001] = 4
#
# LVS 02/06/2018

watpolRaster = gdal.Open(watpol_tif)
if watpolRaster is not None:
    watpolArray = watpolRaster.ReadAsArray()
    diffClassArray[watpolArray == 2] = 2.000
    watpolRaster = None
    watpolArray = None
# LVS 02/06/2018 end

diffClassArray.astype(numpy.int16)

print ("Regroup...")

label = label(diffClassArray, connectivity=2)
regions = regionprops(label)
for region in regions:
    if region.area < 4:
        diffClassArray[region.coords[:, 0], region.coords[:, 1]] = 2

diffClassBand.WriteArray(diffClassArray)
diffClassBand.FlushCache()
diffClass = None

diffArray = None
maskeddensityArray1 = None
maskeddensityArray2 = None

# sys.exit()

### identity line dataset with the reclassified diff raster of line density ###
print ('Identity...')
inDriver = ogr.GetDriverByName("ESRI Shapefile")
nhdDataSource = inDriver.Open(clipped_nhdData, 0)
nhdLayer = nhdDataSource.GetLayer(0)
edsDataSource = inDriver.Open(clipped_edsData, 0)
edsLayer = edsDataSource.GetLayer(0)

srs = nhdLayer.GetSpatialRef()
output1 = outdir + '/{0}_identity.shp'.format(os.path.splitext(os.path.basename(clipped_nhdData))[0])
outIdentity1 = inDriver.CreateDataSource(output1)
outLayer1 = outIdentity1.CreateLayer('outIdentity1.shp', srs, ogr.wkbLineString)
# outLayer1=outIdentity1.CreateLayer('outIdentity1.shp',srs,inLayer1.GetGeomType())

output2 = outdir + '/{0}_identity.shp'.format(os.path.splitext(os.path.basename(clipped_edsData))[0])
srs = edsLayer.GetSpatialRef()
outIdentity2 = inDriver.CreateDataSource(output2)
outLayer2 = outIdentity2.CreateLayer('outIdentity2.shp', srs, ogr.wkbLineString)
# outLayer2=outIdentity2.CreateLayer('outIdentity2.shp',srs,inLayer2.GetGeomType())

inLayerDefn1 = nhdLayer.GetLayerDefn()
for i in range(0, inLayerDefn1.GetFieldCount()):
    fieldDefn = inLayerDefn1.GetFieldDefn(i)
    outLayer1.CreateField(fieldDefn)
outLayer1.CreateField(ogr.FieldDefn("clc_code", ogr.OFTInteger))
outLayerDefn1 = outLayer1.GetLayerDefn()

inLayerDefn2 = edsLayer.GetLayerDefn()
for i in range(0, inLayerDefn2.GetFieldCount()):
    fieldDefn = inLayerDefn2.GetFieldDefn(i)
    outLayer2.CreateField(fieldDefn)
outLayer2.CreateField(ogr.FieldDefn("clc_code", ogr.OFTInteger))
outLayerDefn2 = outLayer2.GetLayerDefn()

step = float(cellsize) / 4

for i_feature in range(0, nhdLayer.GetFeatureCount()):
    inFeature = nhdLayer.GetFeature(i_feature)
    geom = inFeature.GetGeometryRef()
    for i_geometry in range(0, max(1, geom.GetGeometryCount())):
        if i_geometry == 0 and geom.GetGeometryCount() == 0:
            g = geom
        else:
            g = geom.GetGeometryRef(i_geometry)
        pts = []
        for i_point in range(0, g.GetPointCount() - 1):
            start = g.GetPoint_2D(i_point)
            pts += [start]
            end = g.GetPoint_2D(i_point + 1)
            L = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            if L == 0: continue
            u = [(end[0] - start[0]) * step / L, (end[1] - start[1]) * step / L]
            cx = start[0] + u[0]
            cy = start[1] + u[1]
            i_start = int((start[1] - originY) / pixelHeight)
            j_start = int((start[0] - originX) / pixelWidth)
            while ((cx - start[0]) * (cx - end[0]) < 0 or (cy - start[1]) * (cy - end[1]) < 0):
                i = int((cy - originY) / pixelHeight)
                j = int((cx - originX) / pixelWidth)
                # print str(i)+","+str(j)+","+str(i_start)+","+str(j_start)
                if i < rows and j < cols:
                    if diffClassArray[i, j] != diffClassArray[i_start, j_start]:
                        pts += [(cx, cy)]
                        clc = diffClassArray[i_start, j_start]
                        # create a new line
                        outFeature = ogr.Feature(outLayerDefn1)
                        for i_field in range(0, outLayerDefn1.GetFieldCount() - 1):
                            # outFeature.SetField(outLayerDefn1.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
                            outFeature.SetField(i_field, inFeature.GetField(i_field))
                        outFeature.SetField("clc_code", clc)
                        newLine = ogr.Geometry(ogr.wkbLineString)
                        for i_pt in range(0, len(pts)):
                            newLine.AddPoint(pts[i_pt][0], pts[i_pt][1])
                        outFeature.SetGeometry(newLine)
                        outLayer1.CreateFeature(outFeature)
                        outFeature = None
                        pts = []
                        pts += [(cx, cy)]
                        start = (cx, cy)
                        i_start = int((start[1] - originY) / pixelHeight)
                        j_start = int((start[0] - originX) / pixelWidth)
                cx = cx + u[0]
                cy = cy + u[1]
            i_end = int((end[1] - originY) / pixelHeight)
            j_end = int((end[0] - originX) / pixelWidth)
            if i_end < rows and j_end < cols:
                if diffClassArray[i_end, j_end] != diffClassArray[i_start, j_start]:
                    pts += [end]
                    clc = diffClassArray[i_start, j_start]
                    outFeature = ogr.Feature(outLayerDefn1)
                    for i_field in range(0, outLayerDefn1.GetFieldCount() - 1):
                        # outFeature.SetField(outLayerDefn1.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
                        outFeature.SetField(i_field, inFeature.GetField(i_field))
                    outFeature.SetField("clc_code", clc)
                    newLine = ogr.Geometry(ogr.wkbLineString)
                    for i_pt in range(0, len(pts)):
                        newLine.AddPoint(pts[i_pt][0], pts[i_pt][1])
                    outFeature.SetGeometry(newLine)
                    outLayer1.CreateFeature(outFeature)
                    outFeature = None
                    pts = []
        pts += [g.GetPoint_2D(g.GetPointCount() - 1)]
        if len(pts) > 1:
            i_start = int((start[1] - originY) / pixelHeight)
            j_start = int((start[0] - originX) / pixelWidth)
            clc = diffClassArray[i_start, j_start]
            outFeature = ogr.Feature(outLayerDefn1)
            for i_field in range(0, outLayerDefn1.GetFieldCount() - 1):
                # outFeature.SetField(outLayerDefn1.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
                outFeature.SetField(i_field, inFeature.GetField(i_field))
            outFeature.SetField("clc_code", clc)
            newLine = ogr.Geometry(ogr.wkbLineString)
            for i_pt in range(0, len(pts)):
                newLine.AddPoint(pts[i_pt][0], pts[i_pt][1])
            outFeature.SetGeometry(newLine)
            outLayer1.CreateFeature(outFeature)
            outFeature = None

for i_feature in range(0, edsLayer.GetFeatureCount()):
    inFeature = edsLayer.GetFeature(i_feature)
    geom = inFeature.GetGeometryRef()
    for i_geometry in range(0, max(1, geom.GetGeometryCount())):
        if i_geometry == 0 and geom.GetGeometryCount() == 0:
            g = geom
        else:
            g = geom.GetGeometryRef(i_geometry)
        pts = []
        for i_point in range(0, g.GetPointCount() - 1):
            start = g.GetPoint_2D(i_point)
            pts += [start]
            end = g.GetPoint_2D(i_point + 1)
            L = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            if L == 0: continue
            u = [(end[0] - start[0]) * step / L, (end[1] - start[1]) * step / L]
            cx = start[0] + u[0]
            cy = start[1] + u[1]
            i_start = int((start[1] - originY) / pixelHeight)
            j_start = int((start[0] - originX) / pixelWidth)
            while ((cx - start[0]) * (cx - end[0]) < 0 or (cy - start[1]) * (cy - end[1]) < 0):
                i = int((cy - originY) / pixelHeight)
                j = int((cx - originX) / pixelWidth)
                if diffClassArray[i, j] != diffClassArray[i_start, j_start]:
                    pts += [(cx, cy)]
                    clc = diffClassArray[i_start, j_start]
                    # create a new line
                    outFeature = ogr.Feature(outLayerDefn2)
                    for i_field in range(0, outLayerDefn2.GetFieldCount() - 1):
                        # outFeature.SetField(outLayerDefn2.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
                        outFeature.SetField(i_field, inFeature.GetField(i_field))
                    outFeature.SetField("clc_code", clc)
                    newLine = ogr.Geometry(ogr.wkbLineString)
                    for i_pt in range(0, len(pts)):
                        newLine.AddPoint(pts[i_pt][0], pts[i_pt][1])
                    outFeature.SetGeometry(newLine)
                    outLayer2.CreateFeature(outFeature)
                    outFeature = None
                    pts = []
                    pts += [(cx, cy)]
                    start = (cx, cy)
                    i_start = int((start[1] - originY) / pixelHeight)
                    j_start = int((start[0] - originX) / pixelWidth)
                cx = cx + u[0]
                cy = cy + u[1]
            i_end = int((end[1] - originY) / pixelHeight)
            j_end = int((end[0] - originX) / pixelWidth)
            if diffClassArray[i_end, j_end] != diffClassArray[i_start, j_start]:
                pts += [end]
                clc = diffClassArray[i_start, j_start]
                outFeature = ogr.Feature(outLayerDefn2)
                for i_field in range(0, outLayerDefn2.GetFieldCount() - 1):
                    # outFeature.SetField(outLayerDefn2.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
                    outFeature.SetField(i_field, inFeature.GetField(i_field))
                outFeature.SetField("clc_code", clc)
                newLine = ogr.Geometry(ogr.wkbLineString)
                for i_pt in range(0, len(pts)):
                    newLine.AddPoint(pts[i_pt][0], pts[i_pt][1])
                outFeature.SetGeometry(newLine)
                outLayer2.CreateFeature(outFeature)
                outFeature = None
                pts = []
        pts += [g.GetPoint_2D(g.GetPointCount() - 1)]
        if len(pts) > 1:
            i_start = int((start[1] - originY) / pixelHeight)
            j_start = int((start[0] - originX) / pixelWidth)
            clc = diffClassArray[i_start, j_start]
            outFeature = ogr.Feature(outLayerDefn2)
            for i_field in range(0, outLayerDefn2.GetFieldCount() - 1):
                # outFeature.SetField(outLayerDefn2.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
                outFeature.SetField(i_field, inFeature.GetField(i_field))
            outFeature.SetField("clc_code", clc)
            newLine = ogr.Geometry(ogr.wkbLineString)
            for i_pt in range(0, len(pts)):
                newLine.AddPoint(pts[i_pt][0], pts[i_pt][1])
            outFeature.SetGeometry(newLine)
            outLayer2.CreateFeature(outFeature)
            outFeature = None

### calculate total length of different CLC value ###
diffClassArray = None
#
# Set class outside of watershed to -9999
#
diffClass = '{0}/diff_class.tif'.format(outdir)
diffClass_sn = '{0}/diff_class_sn.tif'.format(outdir)
# Set values outside boundary to -9999 for the second smoothed raster dataset.
subprocess.call(
    ["gdalwarp", "-cutline", buffered_basin, "-crop_to_cutline", diffClass, "-of", "GTiff", "-dstnodata", "-9999",
     "-overwrite", diffClass_sn])
#
# Cleanup
#
os.system("gdalmanage delete " + nhdDensity)
os.system("gdalmanage delete " + edsDensity)
os.system("gdalmanage delete " + smdenseRaster1)
os.system("gdalmanage delete " + smdenseRaster2)
os.system("gdalmanage delete " + watpol_tif)
os.system("gdalmanage delete " + watpol_tif2)
os.system("gdalmanage delete " + temp_diff)
os.system("gdalmanage delete " + diffClass)

os.system("gdalmanage rename " + diffClass_sn + " " + diffClass)
#
#
print ("Calculate length of different CLC value...")
# s1=dict(clc1=0,clc2=0,clc3=0,clc4=0)
# s2=dict(clc1=0,clc2=0,clc3=0,clc4=0)
outLayer1.ResetReading()
outLayer2.ResetReading()
s1 = {1: 0, 2: 0, 3: 0, 4: 0}
s2 = {1: 0, 2: 0, 3: 0, 4: 0}
for feature in outLayer1:
    # temp='clc'+str(feature.GetField('clc_code'))
    temp = feature.GetField('clc_code')
    geom = feature.GetGeometryRef()
    s1[temp] += geom.Length()

for feature in outLayer2:
    # temp='clc'+str(feature.GetField('clc_code'))
    temp = feature.GetField('clc_code')
    geom = feature.GetGeometryRef()
    s2[temp] += geom.Length()
    # print geom.Length()
    # print 'clc: {0}   ID: {1}'.format(feature.GetField('clc_code'),feature.GetField(0))
nhdDataSource = None
edsDataSource = None

WriteLineToFile(outfile,
                'clc code and corresponding total length for {0}, clc code : total length'.format(clipped_nhdData))
print (s1)
match1 = s1[1] + s1[2]
WriteLineToFile(outfile, 'total length of matching features in {0}: {1}'.format(clipped_nhdData, match1))
mis1 = s1[3] + s1[4]
WriteLineToFile(outfile, 'total length of mismatching features in {0}: {1}'.format(clipped_nhdData, mis1))
WriteLineToFile(outfile,
                'clc code and corresponding total length for {0}, clc code : total length'.format(clipped_edsData))
print (s2)
match2 = s2[2] + s2[3]
WriteLineToFile(outfile, 'total length of matching features in {0}: {1}'.format(clipped_edsData, match2))
mis2 = s2[1] + s2[4]
WriteLineToFile(outfile, 'total length of mismatching features in {0}: {1}'.format(clipped_edsData, mis2))
WriteLineToFile(outfile, 'total CLC = {}'.format((match1 + match2) / (match1 + mis1 + match2 + mis2)))

# Write to secondary text file.
#WriteLineToFile(txt_results, "subbasin, {}".format(subbasin))
WriteLineToFile(txt_results, "os_tot_clc, {}".format((match1 + match2) / (match1 + mis1 + match2 + mis2)))

### create fishnet ###
fishnetData = outdir + '/fishnet.shp'
nMax = 240
nMin = 200
gridsize = (int)(math.sqrt((abs(xmax - xmin) * abs(ymax - ymin)) / nMin))
gridcount = int(math.ceil(abs(xmax - xmin) / gridsize) * math.ceil(abs(ymax - ymin) / gridsize))

if gridcount > nMax or gridcount < nMin:
    gridsize = math.sqrt((abs(xmax - xmin) * abs(ymax - ymin)) / nMin)
    gridcount = int(math.ceil(abs(xmax - xmin) / gridsize) * math.ceil(abs(ymax - ymin) / gridsize))

WriteLineToFile(outfile, 'Creating fishnet using gridsize = {0} gridcount = {1}'.format(gridsize, gridcount))
fishnet(fishnetData, srs, xmin, xmax, ymin, ymax, gridsize, gridsize)
### calculate CLC value in each fishnet grid ###
print ('Identify lines with fishnet grid...')

srs = outLayer1.GetSpatialRef()
outfish1 = outdir + '/{0}_gridded.shp'.format(os.path.splitext(os.path.basename(clipped_nhdData))[0])
fishIdentity1 = inDriver.CreateDataSource(outfish1)
gridLayer1 = fishIdentity1.CreateLayer('fishIdentity1.shp', srs, ogr.wkbLineString)
outfish2 = outdir + '/{0}_gridded.shp'.format(os.path.splitext(os.path.basename(clipped_edsData))[0])
srs = outLayer2.GetSpatialRef()
fishIdentity2 = inDriver.CreateDataSource(outfish2)
gridLayer2 = fishIdentity2.CreateLayer('fishIdentity2.shp', srs, ogr.wkbLineString)

fishDataSource = inDriver.Open(fishnetData, 0)
fishLayer = fishDataSource.GetLayer()

outLayer1.Identity(fishLayer, gridLayer1,
                   ['SKIP_FAILURES=YES', 'PROMOTE_TO_MULTI=NO', 'KEEP_LOWER_DIMENSION_GEOMETRIES=NO'])
outLayer2.Identity(fishLayer, gridLayer2,
                   ['SKIP_FAILURES=YES', 'PROMOTE_TO_MULTI=NO', 'KEEP_LOWER_DIMENSION_GEOMETRIES=NO'])

outIdentity2 = None
outIdentity1 = None

print ('Calculating CLC value in each grid...')
srs = fishLayer.GetSpatialRef()
outgrid = outdir + '/grid_CLC.shp'
outGridSource = inDriver.CreateDataSource(outgrid)
finalGrid = outGridSource.CreateLayer('grid_CLC.shp', srs, ogr.wkbPolygon)
finalGrid.CreateField(ogr.FieldDefn('fishID', ogr.OFTInteger))
finalGrid.CreateField(ogr.FieldDefn('bm_clc1', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_clc2', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_clc3', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_clc4', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_clc1', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_clc2', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_clc3', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_clc4', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_match', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_omi', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_match', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_commi', ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('full_clc', ogr.OFTReal))
finalGridDefn = finalGrid.GetLayerDefn()
outdict = {}

fishLayer.ResetReading()
gridLayer1.ResetReading()
gridLayer2.ResetReading()
for feature in gridLayer1:
    fishid = feature.GetField('fishID')
    if fishid not in outdict:
        outdict[fishid] = [0.0 for x in range(0, 8)]
    clc = feature.GetField('clc_code')
    geom = feature.GetGeometryRef()
    outdict[fishid][clc - 1] += geom.Length()

for feature in gridLayer2:
    fishid = feature.GetField('fishID')
    if fishid not in outdict:
        outdict[fishid] = [0.0 for x in range(0, 8)]
    clc = feature.GetField('clc_code')
    geom = feature.GetGeometryRef()
    outdict[fishid][clc + 3] += geom.Length()

for feature in fishLayer:
    fishid = feature.GetField('fishID')
    if fishid not in outdict:
        continue
    else:
        geom = feature.GetGeometryRef()
        outFeature = ogr.Feature(finalGridDefn)
        outFeature.SetGeometry(geom)
        outFeature.SetField('fishID', fishid)
        temp = outdict[fishid]
        for i in range(0, 8):
            outFeature.SetField(i + 1, temp[i])
        match1 = temp[0] + temp[1]
        mis1 = temp[2] + temp[3]
        match2 = temp[5] + temp[6]
        mis2 = temp[4] + temp[7]
        outFeature.SetField('bm_match', match1)
        outFeature.SetField('bm_omi', mis1)
        outFeature.SetField('tg_match', match2)
        outFeature.SetField('tg_commi', mis2)
        fullCLC = (match1 + match2) / (match1 + mis1 + match2 + mis2)
        outFeature.SetField('full_clc', fullCLC)
        finalGrid.CreateFeature(outFeature)
        outFeature = None

fishIdentity1 = None
fishIdentity2 = None
fishDataSource = None
outGridSource = None
#
# Output matching elevation derived lines
with fiona.open(output2) as input:
    meta = input.meta
    with fiona.open(match_edsData, 'w',**meta) as output:
        for feature in input:
             if feature['properties']['clc_code'] in (2,3):
                 output.write(feature)

WriteLineToFile(outfile, "Finish time " + str(time.ctime()))
end_time = time.time()
seconds = end_time - start_time
minutes = seconds / 60.0
WriteLineToFile(outfile, "Processing minutes: " + str(minutes))
