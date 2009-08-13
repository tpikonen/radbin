EXTENSION=cradbin.so
EXTOBJS=cradbin.o dorebin.o
CFLAGS+=-I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include/numpy

all: $(EXTENSION)

# ---- Link --------------------------- 
$(EXTENSION): $(EXTOBJS) 
	gcc -shared $^ -o $@

# Not needed because of GNU make automatic rules
#$(EXTOBJS): $(EXTSRCS)
#	gcc  -c $< 

clean:
	-rm -f $(EXTENSION) $(EXTOBJS)
