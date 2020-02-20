Neural Network Models on KERAS.

INSTALLING ANACONDA:

1. First download the Python 2.7 64-Bit (x86) installer from
As it will be a fairly large file, I suggest downloading it directly into your machine.

Commands:
cd to the directory where you wish to download
wget https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh

2. Make the file executable:
Command:
chmod +x Anaconda2-5.2.0-Linux-x86_64.sh

3. Run it:
Command:
./Anaconda2-5.2.0-Linux-x86_64.sh

After the license agreement, it will prompt you for the location to install Anaconda. Do not accept the default - you need to change it to something in /home/local, for example
/home/local/vignesh/NN

That will take a little while to finish.

I would suggest not having the installer prepend the Anaconda install
location to your PATH in your .bashrc. Since Anaconda includes its
own version of many things, doing that may result in problems as its
versions of programs may be used unexpectedly. I suggest only setting
the PATH to include Anaconda when you are using it.

4. Set the correct path
Command:
export PATH=/home/local/vignesh/NN/anaconda2/bin:$PATH

5. Update conda (Run this command)
Command:
conda update anaconda

6. To install keras and tensorflow:
Commands:
conda install tensorflow
conda install keras

USING LENET:

lenet.py for training. Simply run:
python lenet.py

lenet_test.py for testing & Validation:
python lenet_test.py <weights_file>
