SRCDIR := src
INCDIR := include
OUTDIR := build
DEPDIR := $(OUTDIR)/deps
BIN := $(OUTDIR)/scan
LIB := $(OUTDIR)/libscan.so

CUDA_ARCH = 75

CC := nvcc
CFLAGS := -I $(INCDIR) -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Wconversion
OFLAGS := -Xcompiler -march=native -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH} -O3 -Xptxas -O3

.PHONY: build run clean

build: $(BIN)

lib: $(LIB)

run: $(BIN)
	@$(BIN) 402653184 # 1.5 GiB vector

$(BIN): $(DEPDIR)/scan.o $(DEPDIR)/main.o
	$(CC) $(CFLAGS) $(OFLAGS) $^ -o $@

$(DEPDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(DEPDIR)
	$(CC) $(CFLAGS) $(OFLAGS) -c $< -o $@

$(DEPDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(DEPDIR)
	$(CC) $(CFLAGS) $(OFLAGS) -c $< -o $@

$(LIB): $(SRCDIR)/scan.cu
	@mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) $(OFLAGS) -shared -Xcompiler -fPIC $< -o $@

clean:
	@rm -rf $(OUTDIR) *.o *.a
