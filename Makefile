ifeq ($(origin CC),default)
    CC := gcc
endif
CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Iinclude
LDFLAGS ?= -lm

# Detect OS
ifeq ($(OS),Windows_NT)
    EXE := .exe
    RM := del /Q /F
else
    EXE :=
    RM := rm -f
endif

# Core library sources and objects
LIB_SRCS := src/tensor.c src/nn.c src/optim.c src/data.c
LIB_OBJS := $(LIB_SRCS:.c=.o)

# Targets
TEST_AUTOGRAD := test_autograd$(EXE)
TEST_NN       := test_nn$(EXE)
DEMO_XOR      := demo_xor$(EXE)
DEMO_CNN      := demo_cnn$(EXE)
DEMO_TRANS    := demo_transformer$(EXE)

TARGETS := $(TEST_AUTOGRAD) $(TEST_NN) $(DEMO_XOR) $(DEMO_CNN) $(DEMO_TRANS)

.PHONY: all clean test demos

all: $(TARGETS)

src/tensor.o: src/tensor.c include/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

src/nn.o: src/nn.c include/nn.h include/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

src/optim.o: src/optim.c include/optim.h include/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

src/data.o: src/data.c include/data.h include/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_AUTOGRAD): tests/test_autograd.c src/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TEST_NN): tests/test_nn.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(DEMO_XOR): demos/demo_xor.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(DEMO_CNN): demos/demo_cnn.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(DEMO_TRANS): demos/demo_transformer.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test: $(TEST_AUTOGRAD) $(TEST_NN)
	@echo "Running autograd tests..."
	./$(TEST_AUTOGRAD)
	@echo "Running module & optimizer tests..."
	./$(TEST_NN)

demos: $(DEMO_XOR) $(DEMO_CNN) $(DEMO_TRANS)
	@echo "Running XOR demo..."
	./$(DEMO_XOR)
	@echo "Running CNN demo..."
	./$(DEMO_CNN)
	@echo "Running Transformer demo..."
	./$(DEMO_TRANS)

clean:
ifeq ($(OS),Windows_NT)
	-@$(RM) src\*.o tests\*.o demos\*.o $(TARGETS) *.exe *.o 2>nul || exit 0
else
	@$(RM) src/*.o tests/*.o demos/*.o $(TARGETS) *.o
endif
