sudo apt-get update
sudo apt-get install python-pip
pip install tensorflow
pip install networkx
pip install matplotlib
sudo apt-get install openjdk-8-jdk
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
FLOWERS_DATA_DIR=/tmp/flowers-data/
bazel build //inception:download_and_preprocess_flowers
bazel-bin/inception/download_and_preprocess_flowers "${FLOWERS_DATA_DIR}"
bazel build //inception:flowers_train
bazel-bin/inception/flowers_train --num_gpus=1 --batch_size=32 --train_dir=/tmp/flowers_train --data_dir=/tmp/flowers-data
