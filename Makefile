all:: qubo

CFLAGS:= -Wall -funroll-loops -O9
LFLAGS:= -lm

qubo: Makefile qubo.c
	gcc -o qubo qubo.c $(CFLAGS) $(LFLAGS)

.PHONY : clean
clean:
	rm -f qubo
