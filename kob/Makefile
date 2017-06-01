TARGET="./main"
CC=g++
LINK=
OPT=-std=c++11 -g -v
INCLUDES=
LIBS=

SRC=main.cpp
HED=main.h
OBJ=$(SRC:main.cpp=main.o)

all: $(TARGET)

$(TARGET):$(OBJ)
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -o $(TARGET) $(OBJ) $(LINK)

.cpp.o:
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -c $< -o $@

makefile.depend:
	g++ -MM -MG $(SRC) $(OPT) $(LINK) > makefile.depend

clean:
	rm ./main
