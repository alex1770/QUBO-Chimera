all:: dummy-sing
par:: dummy-par

# Long-winded way of ensuring that "make" makes the single-threaded version
# and "make par" makes the multi-threaded version.

CFLAGS:= -Wall -funroll-loops -O9
LFLAGS:= -lm
ifeq ($(filter par,$(MAKECMDGOALS)),)
  CFLAGS:= $(CFLAGS) -Wno-unknown-pragmas
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

.PHONY : clean
clean:
	rm -f qubo dummy-par dummy-sing
