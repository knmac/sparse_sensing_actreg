# clean up
rm -rf build
mkdir build

# build
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
make
