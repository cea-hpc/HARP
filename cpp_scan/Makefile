SRCDIR := src
INCDIR := include
OUTDIR := build
DEPDIR := $(OUTDIR)/deps
BIN := $(OUTDIR)/scan

CUDA_ARCH = 75

CC := nvcc
CFLAGS := -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -march=native -I $(INCDIR)
OFLAGS := -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH} -O3 -Xptxas -O3

.PHONY: build run clean

build: $(BIN)

run: $(BIN)
	@$(BIN) 402653184 # 1.5 GiB vector

$(BIN): $(DEPDIR)/scan.o $(DEPDIR)/main.o
	$(CC) $(CFLAGS) $(OFLAGS) $^ -o $@

$(DEPDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(DEPDIR)
	$(CC) $(CFLAGS) $(OFLAGS) -c $< -o $@

clean:
	@rm -rf $(OUTDIR)
