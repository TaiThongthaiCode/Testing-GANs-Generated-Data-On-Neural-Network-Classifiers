## CS 66 Final Project Lab Notebook

Name 1: Tai Thongtai

Name 2: Sam Yan

Name 3: Richard Chen

userId1: nthongt1

userId2: syan2

userId3: rchen2

Project Title: Simulating Datasets with GANs

---
### Rich, Sam, Tai: 04-28-19 (2hrs)
- Looked up examples of GANS and how it works
- Copied example of MNIST GANs file to be altered for our dataset
    - Credit Link: https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py?fbclid=IwAR3qgjN5ZrcyCruhw3S3IgVd3b02NsIGO-sMi-2dQKLRNTyHZxFVVI7q1pU
- Read over file and tried to interpret usage for our data

### Rich, Sam, Tai: 04-30-19 (2hrs)
- Learned and Researched Adam Optimizer
    - Link: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
- Learned tf.keras.layers.Dense function
    - Link: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
- Learned LeakyReLu and its advantage over ReLu
    - Link: https://www.quora.com/What-are-the-advantages-of-using-Leaky-Rectified-Linear-Units-Leaky-ReLU-over-normal-ReLU-in-deep-learning
- Learned .summary() function
    - Link: https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/

### Sam, Tai, Rich: 05-01-19 (3.5hrs)
- Tried to fine tune the original code final to run cifar-10
- Changed some code and successfully read cifar-10
- Altered parameters to better fit cifar-10
- Result after 10000 epochs:
![99800_old](99800_old.png)

- Image isn't successfully training -- might need to switch to a better format/github repo

### Tai, Sam, Rich: 05-04-19 (2hrs)
- Researched other codes and tried to learn what made other code better

### Rich, Tai, Sam: 05-06-19 (4hrs)
- Overwritten what we had and tuned parameters to run mnist better
- Tried to better understand functions in Keras
- Thinking that we might move to mnist and try to better it since cifar-10 isn't fairing well
- Still tried to run cifar-10

### Tai, Sam, Rich: 05-07-19 (7hrs)
- Since our code isn't working well after many tries and fine-tuning, we decided to change the
original github repo code.
- Researched and tried to find other code
  - Note: Many codes found online have different libraries installed that are different from what is installed
  on our school computer. Finding a well-working GANs program that runs with our school computer took a very long
  time to find
- Eventually found codes that ran and produced both MNIST and cifar-10 datasets

### Tai, Sam, Rich: 05-10-19 (5hrs)
- Tried to understand the code that we git pulled
- Fine-tuned parameters to better our results
- Trial-by-error many of the parameters until the codes ran smoothly on the lab computers

### Sam, Tai, Rich: 05-12-19 (~39.5hrs training; 6hrs understanding/tuning/working)
- Ran cifar-10 all day (for two days) and trained
- Check on how the training was working every few hours
- Sometimes ran into segfaults due to lack of memory.
  - After reading through the code and trying to understand, we found that the code was saving its weights
  and other necessary parameters along with the history of training every epoch. This turned out to be the reason
  why we ran out of memory so fast (even with the virtual machine).
  - We ended up changing code around so it wouldn't save every epoch, but after every few hundred epochs are so. A few parts is
  done by trial and error
- Continued running code.
- We found that the images were "plateau-ing" in terms of better quality, but we continued to train
- Found that we were able to recognize the images from afar... probably because cifar-10 data is 32 pixels by 32 pixels
- Here are some of the results:
  - MNIST: (After 1000 epochs)
  ![60634409_614797952365796_8583153272431312896_n](60634409_614797952365796_8583153272431312896_n)
  - Cifar-10: (After 1000 epochs)
  ![plot_epoch_1000_generated](plot_epoch_1000_generated)

### Tai, Sam, Rich: 05-13-19 (~24hrs training; 2hrs understanding/working)
- Figured out how to save weights
- Figured out how to concatenate data
- Understand some of the code that runs the mnist dataset
- Learned .npy file types
- Trained the dataset for ~24 hrs

### Rich, Tai, Sam: 05-15-19 (9hrs)
- Analyzed both generated cifar10 and mnist datasets
- Hand-labeled mnist dataset (so painstaking) for preprocessing to analyze GANs
- Found that when using generated mnist data to classify, the accuracy rate is ~25%
  - We first predicted that it might be due to human-error (aka 'mis-labeling' each datum)
  - To test this, we shuffled the labels of the generated mnist file so none of the labels
  match the generated image. The result wasn't too much better, eliminating human error
  - Created and finalized the powerpoint presentation
  - Used the generated images of both cifar-10 and mnist
