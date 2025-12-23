# ==============================
# Makefile for CS610 Assignment 4
# Builds Problem 1, Problem 2, Problem 3 (v1–v4), Problem 4
# No object files are kept
# ==============================

NVCC        := nvcc
NVCC_FLAGS  := -O2 -std=c++17 -arch=sm_80 -lineinfo -res-usage -src-in-ptx

# ---------------------------------
# Problem 1
# ---------------------------------
P1DIR       := problem1-dir
SRC_P1      := $(P1DIR)/problem1.cu
EXE_P1      := $(P1DIR)/p1

# ---------------------------------
# Problem 2
# ---------------------------------
P2DIR       := problem2-dir
SRC_P2A     := $(P2DIR)/problem2a.cu
SRC_P2B     := $(P2DIR)/problem2b.cu
EXE_P2A     := $(P2DIR)/p2a
EXE_P2B     := $(P2DIR)/p2b

# ---------------------------------
# Problem 3 (v1–v4)
# ---------------------------------
P3DIR       := problem3-dir
SRC_P3V1    := $(P3DIR)/problem3-v1.cu
SRC_P3V2    := $(P3DIR)/problem3-v2.cu
SRC_P3V3    := $(P3DIR)/problem3-v3.cu
SRC_P3V4    := $(P3DIR)/problem3-v4.cu

EXE_P3V1    := $(P3DIR)/p3_v1
EXE_P3V2    := $(P3DIR)/p3_v2
EXE_P3V3    := $(P3DIR)/p3_v3
EXE_P3V4    := $(P3DIR)/p3_v4

# ---------------------------------
# Problem 4
# ---------------------------------
P4DIR       := problem4-dir
SRC_P4      := $(P4DIR)/problem4.cu
EXE_P4      := $(P4DIR)/p4

# ---------------------------------
# Build EVERYTHING
# ---------------------------------
all: $(EXE_P1) $(EXE_P2A) $(EXE_P2B) \
     $(EXE_P3V1) $(EXE_P3V2) $(EXE_P3V3) $(EXE_P3V4) \
     $(EXE_P4)

# ---------------------------------
# Rules
# ---------------------------------

# Problem 1
$(EXE_P1): $(SRC_P1)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Problem 2
$(EXE_P2A): $(SRC_P2A)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(EXE_P2B): $(SRC_P2B)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Problem 3
$(EXE_P3V1): $(SRC_P3V1)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(EXE_P3V2): $(SRC_P3V2)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(EXE_P3V3): $(SRC_P3V3)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(EXE_P3V4): $(SRC_P3V4)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Problem 4
$(EXE_P4): $(SRC_P4)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ---------------------------------
# Clean
# ---------------------------------
clean:
	rm -f \
	$(EXE_P1) \
	$(EXE_P2A) $(EXE_P2B) \
	$(EXE_P3V1) $(EXE_P3V2) $(EXE_P3V3) $(EXE_P3V4) \
	$(EXE_P4)

.PHONY: all clean
