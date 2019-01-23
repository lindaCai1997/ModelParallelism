# Memory-constrained Device Placement Algorithms
The algorithms can be run on any NN models. We'll be using Inception-V3 as an example.

## File Hierarachy
The algorithms lie in the Simulator folder. The Inception-V3 folder is downloaded from the TensorFlow Github Repository.

    Simulator
        Simulate.py
        m_sct.py
        m_etf.py
        m_topo.py
    Inception-V3
        g3doc
        inception
        README.md
        WORKSPACE
    inception_train.py
    mosek.lic

## Installation
* Basic Dependencies

      pip install tensorflow
      pip install networkx
      pip install matplotlib
      pip install scipy

* Mosek

      pip install cvxopt
      pip install -f https://download.mosek.com/stable/wheel/index.html Mosek --user
      cp mosek.lic ~/.mosek/mosek.lic

* Bazel

      sudo apt-get install openjdk-8-jdk
      echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
      curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
      sudo apt-get update && sudo apt-get install bazel
    
* Flower Dataset

      FLOWERS_DATA_DIR=/tmp/flowers-data/
      bazel build //inception:download_and_preprocess_flowers
      bazel-bin/inception/download_and_preprocess_flowers "${FLOWERS_DATA_DIR}"
      
* Simulator

      cp Simulator/* Inception-V3/inception/
      cp inception_train.py Inception-V3/inception/inception_train.py

## Train
Replace the \<hosts\> with the ip:port of each device seperated by comma. And replace the \<index\> with the task index of the device where 0 indicates the cheif device.
    
    bazel-bin/inception/flowers_train --batch_size=32 --train_dir=/tmp/flowers_train --data_dir=/tmp/flowers-data --hosts=<hosts> --task_index=<index>

