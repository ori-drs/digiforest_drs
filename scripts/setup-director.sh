#!/bin/bash

cd ../..
echo "Installing packages in ${PWD}"
read -p "Press enter to continue"
# pcl-python
echo "Installing python-pcl system wide"
read -p "Press enter to continue"
sudo apt-get install libpcl-dev python3-pip -y
sudo pip3 install Cython numpy==1.23.4 lxml
git clone https://github.com/ori-drs/python-pcl.git
cd python-pcl
python3 setup.py build_ext --inplace
sudo python3 setup.py install
cd -

echo "Setting up DRS PPA"
echo "Please copy the key in the current directory ${PWD}"
read -p "Press enter to continue"
sudo apt-key add drs_packages_server_public.key
sudo add-apt-repository "deb http://drs-packages.robots.ox.ac.uk $(lsb_release -s -c) main"
sudo apt update
sudo apt install ros-noetic-ctk-python-console ros-noetic-python-qt ros-noetic-qt-property-browser ros-noetic-pcl-plugin -y


# director
echo "Cloning director "
echo "Requires numpy <= 1.23.4"
read -p "Press enter to continue"
git clone https://github.com/ori-drs/director_digiforest.git
git clone https://github.com/ori-drs/director.git
git clone https://github.com/ori-drs/vtk_ros.git
git clone https://github.com/ori-drs/cv_utils.git
