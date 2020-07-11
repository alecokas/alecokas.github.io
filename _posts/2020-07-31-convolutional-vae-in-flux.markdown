---
layout: post
title:  "Convolutional VAE in Flux"
date:   2020-07-10 10:00:00 +0000
categories: Julia Flux VAE
---

In this post, we'll have a look at variational autoencoders and demonstrate how to build one for the FashionMNIST dataset using Flux. This content follows on from the previous post where I introduce Flux, a great machine learning framework for Julia.

### Formulating the Variational Autoencoder
Before looking at the implementation, I'll present a short overview of autoencoders and the differentiating features of a variational autoencoder (VAE). For a full description of the background and how the loss function is derived, have a look at the original VAE paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).


#### Autoencoders
An autoencoder is made up of two principle components, the encoder and the decoder. The role of the encoder is to extract learnt features from the input data, $$ x $$, and represent them in a constrained latent space, $$ z $$. Ideally, this latent space, sometimes called the <i>bottleneck</i> layer, is a respresentation of the compressed underlying characteristics of the data. The decoder then generates a reconstruction of the original image, $$ \hat{x} $$, which we want to closely resemble the input data. If the encoder and decoders are modelled using neural networks, we can train the autoencoder to minimise the reconstruction loss between $$ x $$ and $$ \hat{x} $$.

#### Making them Variational
The key difference between the vanilla autoencoder, and a VAE, is in the treatment of the latent space, $$ z $$. For VAEs we model the latent space as a probability distribution, $$ q(z|x) $$, which approximates some prior, $$ p(z) $$. Typically this prior is the Gaussian $$ \mathcal{N} (0, 1)$$. We train the encoder to learn the mean and standard deviation of $$ q(z | x) $$, which we then use to generate samples to feed into the decoder network. Since we still want to train our VAE using backpropagation and gradient descent, we need a mechanism for removing the sampling operation from the backprogation path whilst still obtaining samples of $$z$$. To this end, we apply the reparameterisation trick, and perform our sampling via $$ z \sim \mu + \sigma \odot \epsilon $$ where $$ \epsilon \sim p(z) $$. <br/><br/>
As before, we define our loss function such that we minimise the reconstruction loss between the original $$ x $$ and the reconstruction $$ \hat{x} $$, however we include an additional KL loss term which penalises the model when $$ q(z | x) $$ deviates from $$ p(z) $$. This loss function ends up being equivalent to maximising the Evidence Lower Bound (ELBO). If we parameterise our encoder and decoder with the parameters $$ \phi $$ and $$ \theta $$ respectively, the objective can be written as follows.

$$ \mathcal{L}(\theta, \phi; x, z)  = \mathbb{E}_{q_{\phi}(z|x)}[log(p_{\theta}(x|z))] - D_{KL}(q_{\phi(z|x)}\|p(z))$$

### Building the VAE in Flux

#### Loading the FashionMNIST dataset
FashionMNIST is an incredibly popular benchmarking dataset, made up of low-resolution greyscale images of clothes, which operates as a simple drop-in replacement for the simpler original MNIST dataset. Each image is one of 10 possible clothing item types, which are used as the 10 labelled classes. The full dataset, established by [Zalando](https://research.zalando.com/), is made up of a training set of 60 000 images and a test set of 10 000 images, where all images are 28 by 28 in pixel width. For our demonstration, we zero-pad each image to 32 by 32 pixels so that we can apply a similar model architecture as documented in  [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl) and [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599). <br/>

We can define a function to do just that, and return a <code>Flux.Data.DataLoader</code> object which will handle the batching and shuffling of our data.

{% highlight julia %}
function get_train_loader(batch_size, shuffle::Bool)
    # FashionMNIST is made up of 60k 28 by 28 greyscale images
    train_x, train_y = FashionMNIST.traindata(Float32)
    train_x = reshape(train_x, (28, 28, 1, :))
    train_x = parent(padarray(train_x, Fill(0, (2,2,0,0))))
    return DataLoader(train_x, train_y, batchsize=batch_size, shuffle=shuffle, partial=false)
end
{% endhighlight %}

### Defining the model




### Diving into the code
{% highlight julia %}
function foo()
  println("Foo")
end
{% endhighlight %}
<!-- <div class="language-python highlighter-rouge"><div class="highlight"><pre class="highlight">
<code>
import Statistics
def this():
    pass

</code></pre></div></div> -->

<!-- <div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight">
<code>@article{kastanos20fluxvae,
  title   = "Convolutional VAE in Flux",
  author  = "Alexandros Kastanos",
  journal = "alecokas.github.io",
  year    = "2020",
  url     = "http://127.0.0.1:4000/julia/flux/vae/2020/07/10/convolutional-vae-in-flux.html"
}
</code></pre></div></div> -->