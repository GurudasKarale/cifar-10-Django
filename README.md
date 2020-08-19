# cifar-10-using-Django webapp
 


# Convolutional Neural Network


![Image of arch](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/arch.jpg)
                                            source:google
                                            
Here, i have emphasized more on how backpropagation works in convolutional layer.
To begin with, i would like to give the brief introduction of cnn architecture.

The fundamental and most important part of CNNs are the kernels. They are used to extract specific features from the image.

Kernel is just the matrix of coefficients which are called weights,these weights are trained to detect specific features.

__Convolution__ is an element-wise product and sum between input image and the kernel which denotes whether specific feature is present in the image or not.

__Max pooling__ reduces the size of image keeping its edge information as it is. Simply, it reduces the redundancy to speed  up the training process and reduce amount of memory required. A window is passed over the image and maximum value within the window is pooled in the output matrix.  

__Forward Pass__

In forward pass, image is passed through multiple convolution operations(say two convolution layer) and then its size is reduced using max pooling.
The output matrix formed after max pooling is flattened to form the 1D feature vector which is applied to the fully connected layer or multi layer perceptron network.
Here https://github.com/GurudasKarale/xor_neural_network/blob/master/README.md i have implemented simple exor problem which shows how fully connected network works.

__Backpropagation__

Convolution between the 3x3 matrix and the 2x2 matrix is shown below.

![Image of conv](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/1conv.PNG)

Our objective is to backpropagate the error ,and update the weights accordingly using optimization algorithm like stochastic gradient descent. So firstly , gradient of error with respect to filter weights and gradient of error with respect to input is calculated.

Taking partial derivative of Error w.r.t filter weights:

![Image of diff](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/2df.PNG)

__Substituting  equations A,B,C,D in equation 1 we get__

![Image of conv](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/3dodf.PNG)

![Image of conv](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/4dedfX.PNG)

__Above equations can be represented in the matrix form as follows:

![Image of conv](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/5Matrix.PNG)

Here, we can see that partial derivative of Error w.r.t filter weights depend upon input and the partial derivative of error  w.r.t output.

__Second most important thing is to calculate the partial derivative of error w.r.t input i.e ∂E/∂X.__

![Image of conv](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/7fully.PNG)

So, to calculate dConv1,Full - Convolution of rotated(180 degree) Filter2 and dConv2 is taken.
Here dconv2 is nothing but (∂E/∂X) which acts as an input to calculate dConv1 i.e we are back-propagating the error from end to start.
Eventually, to calculate the dFilter1, dConv1 acts as ∂E/∂O and is convolved with input image.

__dConv2= backPropagate(dPool(∂E/∂X))__

__dF2 = convolution(conv1 , dConv2(∂E/∂O))__ 

__dConv1= full-convolution(dConv2(∂E/∂X)  ,  rotated filter2 )__

__dF1 = convolution(image , dConv1)__

__dImage=  full-convolution(dConv1,rotated filter1)__

 __Calculation of ∂E/∂X  :__
 
 ![Image of conv](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/8dedx.PNG)
 
 ![Image of conv](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/9.PNG)
 
 __Similarly,we can calculate__
 
 ![Image of conv](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/10.PNG)
 
 __Above equations can be written in the matrix form as follows:__
 
 ![Image of conv](https://github.com/GurudasKarale/cifar-10-using-Django/blob/master/img/11.PNG)
 
 __This is how gradients are calculated which are then used by optimization algorithm to update weights.__

