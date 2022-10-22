mex -v CXXFLAGS='$CXXFLAGS -std=c++11 -O3' -I../c++ -largeArrayDims sparse_prod.cpp
mex -v CXXFLAGS='$CXXFLAGS -std=c++11 -O3' -I../c++ -largeArrayDims read_text_mat.cpp
mex -v CXXFLAGS='$CXXFLAGS -std=c++11 -O3' -I../c++ -largeArrayDims write_text_mat.cpp
mex -v CXXFLAGS='$CXXFLAGS -std=c++11 -O3' -I../c++ -largeArrayDims sort_sparse_mat.cpp
mex -v CXXFLAGS='$CXXFLAGS -std=c++11 -O3' -I../c++ -largeArrayDims get_rand_mat.cpp
mex -v CXXFLAGS='$CXXFLAGS -std=c++11 -O3' -I../c++ -largeArrayDims split_data.cpp
mex -v CXXFLAGS='$CXXFLAGS -std=c++11 -O3' -I../c++ -largeArrayDims mat_to_libsvm.cpp
