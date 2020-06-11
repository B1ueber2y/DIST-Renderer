mkdir -p data
cd data
# download shapenet (comment if needed)
wget http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip
unzip ShapeNetCore.v2.zip
rm ShapeNetCore.v2.zip
# download sun2012 (comment if needed)
wget http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz
tar zxf SUN2012pascalformat.tar.gz
rm SUN2012pascalformat.tar.gz
# download view distribution
wget ftp://b1ueber2y.me/DIST/external/view_distribution.tar.gz
tar zxf view_distribution.tar.gz
rm view_distribution.tar.gz
cd ..

# download blender-2.71 (Yes! We found that later version is imcompatible)
mkdir -p install
cd install
wget https://download.blender.org/release/Blender2.71/blender-2.71-linux-glibc211-x86_64.tar.bz2
tar xjf blender-2.71-linux-glibc211-x86_64.tar.bz2
rm blender-2.71-linux-glibc211-x86_64.tar.bz2
cd ..

