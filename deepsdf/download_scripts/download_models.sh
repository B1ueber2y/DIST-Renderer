mkdir -p experiments
wget ftp://b1ueber2y.me/DIST/models.tar.gz
tar zxvf models.tar.gz
mv models/* experiments/
rm -rf models && rm models.tar.gz
