# Maximum-Activation-Using-VGG

The idea of "activation maximization" is to ask the question: what input image would most strongly activate a
particular output class prediction? Specifically, suppose we have an input image x, which could be blank, or
noise, or some other image. We feed this image through the CNN to get class activation y^. We then choose a
particular class i and ask: how should I perturb the pixels in x so as to increase y^ ? Deep learning frameworks i
allow us to compute the gradient of a particular class activation y^ with respect to all image pixels x by simply i
backpropagating. We can iterate this process, performing "gradient ascent" in pixel space.


Accuracy of the classifier on the 10000 test images is 85.6% using VGG11
