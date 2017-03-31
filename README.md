Self-Driving Car Engineer Nanodegree

First i just plot 10 random pictures. They seem a bit odd, they have a weird shade to them. Then I plot a histogram showing the distributon of the classes. As you can tell they are fairly uneven. Which makes sense because not all signs in the real world are evenyl distributed.
Number of training examples = 34799
Number of testing examples = 12630


2.To preprocess the data, I grayscaled it, and then normalized it between -1 and 1. I tried smaller values like -.5, and .5, but it seemed to not do as well as the final solution.
I then created new data by using the functions in cell block 12. These functions took in an image and rotated, translated, sheared, and brightened the images. These were all used in the baseline model so i thought it would be a good starting point to increase the accuracy.
From the histogram above in cell 17, they data is more vast. We now have 46714 examples. In the baseline model provided I believe it used 120 thousand+ examples.
Upon further research I found an interesting method called spatial transformers, which I think a team at Google came up with. From what I understood, implementing that inside the network requires less data examples than the models I tried.
    
    
The network has 3 convolutional layers, one fully connected, and the final output layer. Each layer is fed to the next but also fed directly into the fourth layer: fully connected. I did not experiment with the abtch size. The layers were pooled and flattened and combined to make the input into fully connected layer. I think the imput size for the fully connected layer might be too high. Im not really sure.
5x5 conv
5x5 conv
5x5 conv
dropout
fully connected
dropout
classification


Model saved
For Training:
Batch Size: 100
Epochs: 40
Learning rate: .0009
Optimizer: AdamOptimizer
I tried different batch sizes and epochs. I settled on 100 and 40 becuase I got the best result from that however i think 40 might not have been high enough, but it just kind of bumped around between 96 and 97 accuracy until around 35, it continued to increase so I am not really sure how high it could have gone with more epochs.
The adamoptimizer was used in the previous lesson, and upon further research I read nothing but good things about it. I wanted to try a lower learning rate than the original .001, and it helped a little bit.


Test Accuracy: 0.9603325128555298
I started with the LeNet architecture from the previous lessson. This accuracy was not high enough. We were supposed to get the highest accuracy as we could. This architecture is based off of the baseline paper. it consists of three convolutional layers and the final logits layer. its sort of similar to a resnet architecture, based on how they feed layers to the next convolution and also straight to the classifier. its not nearly as deep. They called it multiscale cnn in the paper. I did my best to implement it. They gave a few tips for pooling again after the initial pools of the layers to pass into the fully connected.
I tried histogram equalization because I read a paper written by the winner of the traffic sign classification challenge. They said they used that technique, however it did not help my solution.
I used two dropout layers because I did not want to overfit the data.
i really want to improve upon this though. I did tons of reading into all different methods, and found this all very exciting. i would like to in the future try batch normalization. before this project i had never implemented a cnn on my own before besides in the previous lesson. One i was intrigued by was a cnn with a spatial transformer inside the network. I was not able to fully understand how these work in the short amount of time since i started the project but hope to do further reserch to see if they are actually useful. I believe researchers at Google were the ones to release the paper i read. I also believe in the baseline paper they did pre-training. Not sure if that is true. What i gather is that you train the model on some data first and then train it again on more data. Had i more time and more computational resources, I would have loved to try all different architectures. Looking forward to all critiques of my network.
Validation Accuracy: best was 97.8%
Test Accuracy: 96 %

The images are very blurry, hopefully the preporcessing will make them more readeable.
Predict the Sign Type for Each Image

Test Set Accuracy = 0.600
60% accurate on the pictures i got from the internet.
Output Top 5 Softmax Probabilities For Each Image Found on the Web

Image 0 predictions [9 5 3] percentage [ 0.95289975  0.03721778  0.00455593] actual 3
Image 1 predictions [34 38 35] percentage [  9.98222768e-01   1.77724240e-03   4.40095397e-16] actual 34
Image 2 predictions [ 0 24 25] percentage [  9.99997973e-01   1.59818251e-06   4.46959319e-07] actual 25
Image 3 predictions [14  3  1] percentage [  1.00000000e+00   5.57395276e-13   1.80260351e-14] actual 14
Image 4 predictions [13  0  1] percentage [ 1.  0.  0.] actual 13
The images it got wrong were the first and third. The third image looks nothing like the guess but it was 46 percent sure it was whaat it actually was. Same with the first image, it got the third guess right. This could be because the image is not as high of quality as it should be but more than likely the model is not as well rounded as i would like it be. It is not as good on 'real world data'. again, looking forward to any comments. I realy want to have more knowledge on this subject. I welcome the crituiques.

