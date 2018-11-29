GCC ?= g++

# Gdal
#GDALDIR = /sw/geosoft/gdal-2.0.2-fgdb
#GDALLIB := -L$(GDALDIR)/lib -lgdal
GDALLIB := -lgdal

# Target rules
all: LineDensity
LineDensity.o: LineDensity.c
	$(GCC) -o $@ -c $<
density.o: density.c
	$(GCC) -o $@ -c $<
io.o: io.c
#	$(GCC) -I. -I$(GDALDIR)/include -o $@ -c $<
	$(GCC) -I. -o $@ -c $<
LineDensity: LineDensity.o io.o density.o
	$(GCC) -o ../$@ $+ $(GDALLIB) 
clean: 
	rm -f *.o ../LineDensity

