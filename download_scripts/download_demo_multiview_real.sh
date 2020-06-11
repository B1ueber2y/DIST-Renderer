# init folder
mkdir -p data

# download real-world data for multi view inference
cd data
wget ftp://b1ueber2y.me/DIST/demo_multiview_real.tar.gz
tar zxvf demo_multiview_real.tar.gz
rm demo_multiview_real.tar.gz
cd ..
