TARGET="./build/demo"
CC=g++
LINK=-lTH -lTHNN
OPT=-std=c++11 -g -v
INCLUDES=-I/usr/local/cpp-torch/include -I/usr/local/cpp-torch/include/TH -I/usr/local/cpp-torch/include/THNN
LIBS=-L/usr/local/cpp-torch/lib

SRC=$(shell ls src/*.cpp)
HED=$(shell ls include/*.hpp)
OBJ=$(SRC:main.cpp=main.o)

all: $(TARGET)
$(TARGET):$(OBJ)
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -o $(TARGET) $(OBJ) $(LINK)

.cpp.o:
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -c $< -o $@

makefile.depend:
	g++ -MM -MG $(SRC) $(OPT) $(LINK) > makefile.depend

clean:
	rm ./src/*.o ./makefile.depend
	rm ./build/demo

-include makefile.depend
