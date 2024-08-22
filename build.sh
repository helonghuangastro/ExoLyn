mv src/sol.f90 ./
f2py -m fmatrixsol -c sol.f90 --build-dir src
mv sol.f90 src
mv *.so src
rm -r src/src*
rm src/*.mod
rm -r src/bbdir
rm src/fmatrixsol-f2pywrappers2.f90
rm src/fmatrixsolmodule.c
rm src/meson.build
rm src/sol.o
mv src/fmatrixsol.*.so src/fmatrixsol.so
