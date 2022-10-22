INC=-I../Tools/c++
CXXFLAGS=-std=c++11 -O3 -g
LIBFLAGS=-pthread

all: clean parabel_train parabel_predict

parabel_train:
	$(CXX) -o parabel_train $(CXXFLAGS) $(INC) parabel_train.cpp parabel.cpp $(LIBFLAGS)

parabel_predict:
	$(CXX) -o parabel_predict $(CXXFLAGS) $(INC) parabel_predict.cpp parabel.cpp $(LIBFLAGS)

clean:
	rm -f parabel_train parabel_predict

