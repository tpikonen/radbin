EXTENSION=cyradbin.so
EXTOBJS=cyradbin.o dorebin.o binsearch.o
PYXSRCS=cyradbin.pyx
PYXPRODS=cyradbin.c
PYTHON_INC=$(shell python-config --includes)
CFLAGS+=-shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
 ${PYTHON_INC}

all: $(EXTENSION) test

make_table.o:
	gcc $(CFLAGS) -DMAKE_TABLE -c dorebin.c -o make_table.o

$(EXTENSION): $(EXTOBJS) make_table.o
	gcc $(CFLAGS) $^ -o $@

$(PYXPRODS): $(PYXSRCS)
	cython $<

clean:
	-rm -f $(EXTENSION) $(EXTOBJS) $(PYXPRODS) *.pyc make_table.o

test:
	nosetests

.PHONY: clean test
