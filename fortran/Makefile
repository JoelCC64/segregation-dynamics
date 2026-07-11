FC     = gfortran
FFLAGS = -O2 -std=f95 -Wall -Wextra
DBGFLAGS = -g -O0 -std=f95 -Wall -Wextra -fcheck=all -fbacktrace

all: segregation

segregation: segregation.f90
	$(FC) $(FFLAGS) -o $@ $<

debug: segregation.f90
	$(FC) $(DBGFLAGS) -o segregation $<

clean:
	rm -f segregation *.o *.mod *.dat

.PHONY: all debug clean
