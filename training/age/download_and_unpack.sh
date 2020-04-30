# chmod +x download_and_unpack.sh
curl -o utkface.tar.xz --create-dirs https://static.mnfst.com/models/datasets/utkface.tar.xz
tar xf utkface.tar.xz
curl -o utkface_test.csv --create-dirs https://static.mnfst.com/models/datasets/utkface_test.csv
curl -o utkface_train.csv --create-dirs https://static.mnfst.com/models/datasets/utkface_train.csv
