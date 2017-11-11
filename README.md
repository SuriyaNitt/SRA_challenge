Note: This project is only tested on Ubuntu 16.04

Pipeline Info can be found here: https://suriyanitt.github.io/imageinpainting.html

Use the tag v1.1 for tested code

How To Install:
--------------

There are 5 scripts to aid installation

1. install_deps.sh - installs dependencies listed down in the Dependencies section
2. install_opencv.sh - installs opencv locally inside the project folder
3. install_caffe_enet.sh - builds modified version of caffe for ENet
4. install_application.sh - builds the challenge application
5. install_all.sh - installs all the above

Assumptions:
-----------

1. The person of interest appears upright in the image.
2. The scene is not too crowded so that there is sufficient gap between two person.

Limitations:
-----------

1. Does not work with humans looking small ( this can be improved ).
2. Does not work well with humans in scenes that are not street like.

How to use:
----------

The executable will be placed inside the bin folder.
Navigate to bin folder and run the app as 

./intelligent_inpainting ../test_imgs/sample5.jpg

Click on the human to be removed from the scene.

References:
----------

1. Centrist: A Visual Descriptor for Scene Categorization: https://smartech.gatech.edu/bitstream/handle/1853/31468/09-05.pdf
2. Real-time Human Detection using contour cues: http://c2inet.sce.ntu.edu.sg/Jianxin/paper/ICRA_final.pdf
3. ENet: A Deep Neural Network Architecture for Real-time Semantic Segmentaion: https://arxiv.org/pdf/1606.02147.pdf
4. Region filling and Object Removal by Exemplar-Based Image Inpainting: http://www.irisa.fr/vista/Papers/2004_ip_criminisi.pdf

Dependencies:
------------

1. boost - sudo apt install libboost1.58-dev (change to the version that is already available to your package manager)
2. boost-system - sudo apt install libboost-system1.58-dev (change to the version that is already available to your package manager)
3. boost-thread - sudo apt install libboost-thread1.58-dev (change to the version that is already available to your package manager)
4. boost-filesystem - sudo apt install libboost-filesystem1.58-dev (change to the version that is already available to your package manager)
5. protobuf - sudo apt install libprotobuf-dev
6. gflags - sudo apt install libgflags-dev
7. glog - sudo apt install libgoogle-glog-dev
8. protobuf-compiler - sudo apt install protobuf-compiler
9. hdf5 - sudo apt install libhdf5-dev
10. lmdb - sudo apt install liblmdb-dev
11. LevelDB - sudo apt install libleveldb-dev
12. snappy - sudo apt install libsnappy-dev
13. atlas - sudo apt install libatlas-base-dev
14. gtk2.0 - sudo apt install libgtk2.0-dev
