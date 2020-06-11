# init folder
mkdir -p data

# download PMO data for multi view inference
cd data
wget ftp://b1ueber2y.me/DIST/demo_multiview_syn.tar.gz
tar zxvf demo_multiview_syn.tar.gz
rm demo_multiview_syn.tar.gz
cd ..

