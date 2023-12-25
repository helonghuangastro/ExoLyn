f2py -m fmatrixsol -c src/sol.f90 --build-dir src
mv *.so src
rm -r src/src*
rm src/*.mod
