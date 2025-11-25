# Author : Ruben Rehal | Date : November 2025
PROJECT=orion
BUILD_DIR=build
SRC=$(shell find src -name '*.cpp')
INC=-Iinclude
CXX=arm-none-eabi-g++
AR=arm-none-eabi-ar
CXXFLAGS=-Os -std=c++17 -ffast-math -fno-exceptions -fno-rtti -Wall -Wextra -Werror $(INC)
LDFLAGS=-specs=nosys.specs
LIB=$(BUILD_DIR)/lib$(PROJECT).a

all: build

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB): $(BUILD_DIR) $(SRC)
	$(CXX) $(CXXFLAGS) -c $(SRC)
	$(AR) rcs $(LIB) *.o
	mv *.o $(BUILD_DIR)

build: $(LIB)
	@echo "Built $(PROJECT) archive"

flash: build
	@if [ -z "$(PORT)" ]; then echo "Set PORT=/dev/..."; exit 1; fi
	arm-none-eabi-gdb -ex "target extended-remote $(PORT)" -ex "load" -ex "quit" $(BUILD_DIR)/$(PROJECT).elf

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all build flash clean
