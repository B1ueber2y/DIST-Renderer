# init folder
mkdir -p data

# download testcase for single view
cd data
wget ftp://b1ueber2y.me/DIST/demo_singleview_syn.tar.gz
tar zxvf demo_singleview_syn.tar.gz
rm demo_singleview_syn.tar.gz
cd ..
