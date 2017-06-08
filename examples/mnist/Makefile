TARGET="./demo"
CC=g++
LINK=-lTH -lTHNN /usr/local/lib/libgflags.a -lhdf5 -lhdf5_cpp
OPT=-std=c++11 -g -v
INCLUDES=-I/Users/Andrew/Developer/kob-setup/include -I/Users/Andrew/Developer/kob-setup/include/TH -I/Users/Andrew/Developer/kob-setup/include/THNN -I/Users/Andrew/Developer/myhdf5stuff/dist/HDF5-1.10.1-Darwin/HDF_Group/HDF5/1.10.1/include
LIBS=-L/Users/Andrew/Developer/kob-setup/lib -L/Users/Andrew/Developer/myhdf5stuff/dist/HDF5-1.10.1-Darwin/HDF_Group/HDF5/1.10.1/lib

SRC=$(shell ls ../../kob/*.cpp) main.cpp
OBJ=$(SRC:main.cpp=main.o)

all: $(TARGET)
$(TARGET):$(OBJ)
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -o $(TARGET) $(OBJ) $(LINK)

.cpp.o:
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -c $< -o $@

makefile.depend:
	g++ -MM -MG $(SRC) $(OPT) $(LINK) > makefile.depend

clean:
	rm ./demo

-include makefile.depend
