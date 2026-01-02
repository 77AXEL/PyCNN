CC = gcc

ifeq ($(OS),Windows_NT)
    EXT = dll
    RM = del /Q
else
    UNAME_S := $(shell uname -s)
    RM = rm -f    
    ifeq ($(UNAME_S),Darwin)
        EXT = dylib
    else
        EXT = so
    endif
endif

ifdef GITHUB_ACTIONS
    CFLAGS = -O3 -shared -fPIC -funroll-loops
else
    CFLAGS = -O3 -shared -fPIC -march=native -ffast-math -funroll-loops
endif

TARGET = optimized.$(EXT)
SRC = optimized.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	$(RM) $(TARGET)