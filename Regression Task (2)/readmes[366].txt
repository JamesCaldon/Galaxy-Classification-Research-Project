%%%%%% README FILE  %%%%%

2020/3/8

This is the details for the production of galaxy images.
This email includes the code files (code-es.dir.tar.gz).

The purpose of this task is to find the sizes (Rd) and the mass fractions (fd)
of disks in elliptical (early-type) galaxies using deep learning.

The codes for the CNN (Convolutional neural network) training
and for the testing are test26a.py and test27a.py, respectively.


This is for regression problem for finding two parameter values
(Rd and fd).


(0) Preparation

0.1  Please install a linux system (e.g., Ubuntu), gfortran, Keras/Tensorflow,
and python. 

0.2 Create the following three directories below your home directory on your PC;

runes1.dir
runspt.dir
work.dir

(I)  Image generation for a particular type of galaxies  (in this case,
elliptical galaxies with inner disks) and testing/training CNNs.

(1) Please gunzip and tar xvf the file code-es.dir.tar.gz

This includes all codes. Please copy all files in this code.dir to

runes1.dir

and then compile the code

./comp

(2) You can try to run the shell script "es.sh" that can generate many
barred galaxies using it.  Please do this in runes1.dir ;

./es.sh &

It will take 50-80 minutes to complete the calculation for 1000 images.

(3) Data collection

Data is placed under work.dir as "m1.dir.tar.gz".

This includes training (and testing) data:


Training data

2dft.dat:  (50 x 50 mesh)  x 1000 images
2dftn1.dat:  fd  (normalized to 0 to 10) x 1000 (label data, "Cor1", correct one)
2dftn2.dat:  Rd (normalized to 0 to 10) x 1000 (label data, "Cor2")


The default parameter set is for 1000 images. So you have to change
the nmodel in your python codes  for this when training/testing
with different N images.

Each image has 50 x 50 pixels. The file 2dft.dat has 1000 x (50 x 50) lines
like this;

0.1
0.2
0.3
.
.
.
.
.
0.12

The first 2500 lines are for the first one image, and the next 2500 lines
are for the 2nd  etc etc. Each value corresponds to stellar density at
each pixel.

(4) Training CNN with test26a.py.

You can run this code for a given epoch (after you change the parameter
nmodel = 1000).


(50 Testing CNN with test27a.py.


Testing  data
2dfv.dat (the same as 2dft.dat)
2dfvn.dat fd and Rd (Correct value, "Cor1" and "Cor2").

2dfvn.dat format is as follows (the same as 2dftn1,2.dat, but combined)

1000
1.0 2.0
1.0 2.0
.
.
.


The first and second columns are Cor1 and Cor2 respectively.
You again consistently set nmodel to be 1000 for this 1000 images.
Then run the test27a.py, which will output the predicted theta and phi in  "test27.out", which is  the same format as 2dfvn.dat.
The first and second column from test27.out are predicted fd ("Pre1") and
Rd ("Pre2"), respectively.

Compare the test27.out with 2dfvn.dat to confirm that 
the CNN can predict correctly.

For example, you can plot the predicted values (y-axis) as a function
of the correct ones (x-axis) for the two variables [Figure 1]:
Y-axis = Pre1 (or Pre2), X-axis=Cor 1 (or Cor 2).

You use 0<<x,y<10 for these plots, because the two variables are normalised as 0 - 10. Also plot x=y to show that
the predicted value = correct one.


Also, you can estimate the cosine distance between the two vectors,
(Cor1, Cor2) and (Pre1, and Pre2), and then plot the distribution
of cos distance (0<cos distance<1); divide the cos distance bin into
10 and count the number of images for each bin [Figure 2].

Soon after you finish these tasks, please send me the 
above Figs 1 and 2.

(II) 80% (training) vs 20% (testing) split and training/testing CNNs.

Next step is to split the original data into 80:20 for training/testing
(standard way to test CNNs) and test CNNs. This time, Figure 1 and 2 should be for
20% data only.


(III) Image plot

Please try to plot 50 x 50 images to understand how galaxies look like.
Please send me some figures of the images.
