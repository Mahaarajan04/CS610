# Compiler and flags
CXX := g++

# Files
TARGET := prob_3
SRC := 220600.cpp

# Default build target
all: $(TARGET)

LIBS=-pthread

# Build rule
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
