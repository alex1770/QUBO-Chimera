all:: dummy-sing plant
par:: dummy-par

# Dummy files are a workaround to ensure that "make" makes the
# single-threaded version and "make par" makes the multi-threaded version,
# and compilation is only done when necessary.

CFLAGS:= -Wall -O9 -funroll-loops
LFLAGS:= -lm
ifeq ($(filter par,$(MAKECMDGOALS)),)
  DUM0:= dummy-par
  DUM1:= dummy-sing
else
  CFLAGS:= $(CFLAGS) -fopenmp -DPARALLEL
  DUM0:= dummy-sing
  DUM1:= dummy-par
endif

$(DUM1): Makefile qubo.c $(DUM0)
	gcc -o qubo qubo.c $(CFLAGS) $(LFLAGS)
	touch $(DUM1)

$(DUM0):
	touch $(DUM0)

plant: Makefile plant.c
	gcc -o plant plant.c $(CFLAGS) $(LFLAGS)

.PHONY : clean
clean:
	rm -f qubo dummy-par dummy-sing
