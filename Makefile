IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17
BINDIR=bin
SRCDIR=src
SRC=$(SRCDIR)/nbodysimulation.cu
TARGET=$(BINDIR)/nbodysimulation

.PHONY: clean build run

build: $(SRC)
	mkdir -p $(BINDIR)
	$(COMPILER) $(COMPILER_FLAGS) $(SRC) -o $(TARGET) -Wno-deprecated-gpu-targets

run: build
	./$(TARGET)

clean:
	rm -rf $(BINDIR)
	rm -rf nbody_proof_of_execution.csv
	rm -rf nbody.bin

all: clean build run
