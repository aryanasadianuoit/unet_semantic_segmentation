# unet_semantic_segmentation
 <h2>Table of Contents:</h2>
  <ul>
  <li><a href="#unet"><b>U-Net Architecture</b></a></li>
  <li><a href="#dataset"><b>Dataset</b></a></li>
  </ul>

Semantic segmentation with U-Net model on InteractiveSegmentation dataset (PyTorch)
<section>
<h2 id="unet">U-Net</h2>
<p text-align: justify>U-Net is a neural model which initially has been proposed for biomedical image segmentation. This model is a <b>fully-convolutional network (FCN)</b>, i.e., there is no fully-connected layer at the end similar to classification models. This modification makes this network an appropriate choice for semantic segmentation, where instead of classifying the whole image in different classes, U-Net can classify the level of pixels. The following figure illustrates the U-Net architecture. The model's scheme clarifies the meaning of the name of this model as the model follows a U-pattern.</p>
<br>
  <img src="unet_scheme.png" alt="U-Net architecture">
  <br>
<br>
</section>
<section>
U-Net is created from two main modules:
<br>
 <ptext-align: justify>
<b>Contratcting path</b> (left side of the figure), which its main task is to exploit the features of the input images. This path is created from multiple stacks of [conventional 2D convolutional layers, batch normalization layers, and ReLU activation function]. Note that the original paper does not use batch normalization. This modification has been done to have a more stable training independent of the initialization values. After each two consecutive convolutional stacks, a max-pooling layer with kernel and stride of 2 halves the spatial size of the input image. As the figure shows each convolutional stack, i.e., each row on the left side, increases the number of dimensions while at the same time, reduces the spatial size by using the kernel (3,3). In our modification, for more compatibility with input images with different sizes, we have considered a padding of 1, i.e., the spatial size of the image after each two consecutive convolutional layers is still the same as input. This is helpful for concatenation in the expnasive path, to reproduce the same size image as the output.</p>  
<br>
  <ptext-align: justify>
<b>Expansive path</b>, The model tries to recreate the image, and apply the learned mask for semantic segmentation. In each row of the expansive path, first, a feature representation from the equivalent level in the contracting path is concatenated through the channels dimension. for increasing the spatial size, the model uses 2D transposed convolutions with kernel and stride size of 2, to double the height and width.
  </p>
</section>
<br>
 <section>
 <h2 id="dataset">Dataset</h2>
  
  <ptext-align: justify>
For this repository, we have used <a href="https://www.kaggle.com/4quant/interactivesegmentation">Interactive Segmentation</a> dataset on Kaggle. The dataset contains 140 RGB images with their gray-scale masks. For cheaper computation, we resided the images to (240,240).  We also randomly divided the whole dataset into 0.75, 0.15, and 0.15 fractions of the whole as train, validation, and test set respectively.
</p>
</section>
