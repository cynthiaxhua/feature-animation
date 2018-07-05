# feature-animation
feature recognition and animation for hand-drawn images 

# Feature Extraction: Contour Finding

This technique takes advantage of the fact that features on a face are typically drawn separately, or at the very least using separate strokes. We identify potential contours (continuous curve segments) in an image using OpenCV and apply a version of non-max suppression. The largest identified contour is labeled as the facial outline and eliminated. Each remaining contour is used to calculate a bounding box, corresponding to the four facial features (two eyes, nose, mouth). The identified features are output as cropped pngs that can then be vectorized and fed into the rest of the pipeline. 

Results: Contour Finding was successful at identifying distinct features. In a manual review of the output from 25 test input images, the program produced clean output images of all four individual features in 100% of cases. The advantage of identifying features this way is that it does not use spatial heuristics and so can be applied to abstracted facial images (ex: cubist) or varying poses. However, the disadvantage of this approach is that it will be less successful if facial features are connected or overlapping. In particular, connections between the nose and eyes will cause the two to be grouped together in an output feature. 

![alt text](https://github.com/cynthiaxhua/feature-animation/blob/master/contour_finder.png "Contour Finder")

# Feature Extraction: Windowing

We use a windowing algorithm, where we extract a window of a set shape by sliding over the image, converted it to a pen-stroke format using our conversion algorithm, and finally classified it as a facial feature with a confidence score.

Results: The Classifier had lower accuracy (69.5%) when tested on bitmap to vector converted images. In particular, the classifier mis-categorized 45% of ears as noses based on vectorized images. This is likely because some curvature detail is lost in the vectorization process due to noise or over-simplification of lines; ear images with reduced curvature may begin to resemble nose images. Comparing stroke-based inputs versus vectorized bitmap inputs to the classifier, vectorization adds a 29.5% error rate. 

# Feature Classification

This pipeline was fitted with varying machine learning methods for classification. We compare a bi-directional RNN (with a softmax classification layer) with K-means, K-Nearest Neighbors and a CNN — all implemented using Keras.

![alt text](https://github.com/cynthiaxhua/feature-animation/blob/master/accuracy.png "Classification Accuracy")


# Feature Animation

Our primary approach to animation involves moving through the learned feature space of a variational autoencoder. The goal of image generation for animation is to smoothly move from one image to a different image (for example from an open mouth to a closed one). Autoencoders are suitable here because they learn a to compress input data into a latent space representation, and we propose interpolating between different vectors in this latent space to generate a series of frame by frame moving from the first image to the second. In other words, we are using the learned organization of the feature space to generate intermediate images that when strung together should create a smooth animation. 

![alt text](https://github.com/cynthiaxhua/feature-animation/blob/master/fbf.png "Generated Frame by Frames")

To perform interpolation on our trained VAE, we begin with an identified feature image and pass it to the encoder to generate z0. We randomly choose another feature image from the same class from the Quickdraw dataset and encode it into z1. Then, we create a series of intermediary latent vectors by performing spherical interpolation: 

![alt text](https://github.com/cynthiaxhua/feature-animation/blob/master/interpolation.png "Interpolation Equation")

where t is a fraction that expresses how far in between we interpolate, and α = cos−1(z0 · z1). Typically, 10 interpolations was enough to generate a visu- ally smooth sequence based on qualitative evaluation. In 25 tests, all variations on the two input images successfully pro- duced interpolated images that were fully formed images immediately recognizable as being of the same class. 

To explore the VAE’s learned latent space, we can use t-distributed stochastic neighbor embedding (TSNE), which is a technique for reducing high- dimensionality vectors down to a lower dimensional space. The TSNE allows us to make some observa- tions on the network’s learning. Specifically, our 512-element latent vector z is reduced down to 2 dimensions for x-y chart plotting. TSNE graphs using encoded latent vectors on 200 images show that the VAE successfully created separate clusters for the four classes in its latent space. The TSNE suggests latent vectors for sharp noses were clustered separately from latent vectors for curved noses (observe the rightmost cluster of sharp noses and the cluster curved noses further left). Additionally, it shows that the mouths and eyes are clustered relatively close together, which aligns with their lower accuracy rates in other classification methods we try in Experiments.

![alt text](https://github.com/cynthiaxhua/feature-animation/blob/master/tsne.png "TSNE Diagrams")

A completed animation video generated using classification and interpolation can be seen at https://vimeo.com/261259579. The interpolations were compiled into video at 5 frames per second, and the demo cycles through interpolations between 7 im- ages per feature. Vectorization was not included in this pipeline due to its low performance.

