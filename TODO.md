# TODO
- Train at least one for sign shape (diamond, rectangle, round, triangle, etc.)
- Train at least one for sign type (stop, speed, warning, parking, etc.)
- Create a 5 pages (max) report
  - Machine learning algorithms that you considered
  - Why you selected these approaches
  - Evaluations of the performance of trained model(s)
  - Your ultimate judgement with supporting analysis and evidence
  - Independent evalutation: find more photos of signs and see if it can classify them correctly
    - We should not be re-training the model for this step, it's a "real world" test

## Model options
A bunch of different neural network designs probably:
- Some with different arrangements of convolutional layers/kernals/pooling/etc.
- Some with translations/scaling to prevent overfitting

Spec says "you are required to fully train your own algorithms", but I think that means we could still use one like VGG, GoogLeNet, or ResNet.

## Good links
- Classification - [https://www.tensorflow.org/tutorials/images/classification](https://www.tensorflow.org/tutorials/images/classification)
- Randomly warp/scale images - [https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- **Popular neural network setups:**
  - VGG - [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
  - GoogLeNet - [https://towardsdatascience.com/deep-learning-googlenet-explained-de8861c82765](https://towardsdatascience.com/deep-learning-googlenet-explained-de8861c82765)
  - ResNet - [https://www.mygreatlearning.com/blog/resnet/](https://www.mygreatlearning.com/blog/resnet/)
