---
layout: post
title:  "Flux: The Flexible Machine Learning Framework for Julia"
date:   2020-07-09 10:00:00 +0000
categories: Julia Flux
---

<div style="text-align: justify">
With Julia having established itself as the 6th most loved language in this year's <a href="https://insights.stackoverflow.com/survey/2020#technology-most-loved-dreaded-and-wanted-languages">Stack Overflow Developer survey</a> and <a href="https://juliacon.org/2020/">JuliaCon 2020</a> kicking off in the next couple weeks, I thought this might be a good time to talk about machine learning in Julia. In this post, We'll touch on Julia and some of its more interesting features. Thereafter we'll introduce Flux, a pure-Julia machine learning framework, and compare simple MNIST classifier in Flux to the equivalent Pytorch and Tensorflow 2 implementations.
</div>
<br/>

<div style="text-align:center"><img src="/post_pdfs/machine_learning_in_julia/flux_logo.png" />
</div><br/>

### The Julia Language
<div style="text-align: justify"><p>
The founders of Julia were looking to create a language which was geared towards interactive scientific computing, whilst at the same time supporting more advanced software engineering processes via directo JIT compilation to native code. To facilitate this dual purpose, Julia is dyamically typed with the default behaviour being to allow values of any type when no type specification is provided. Additional expressiveness is permissible via optional typing which provides some of the efficiency gains typically associated with static languages. Having spent my formative programming years writing C++ code, and now almost exclusively working in Python, the option to type my machine learning code is incredibly appealing.
</p>
<p>
The REPL interactive prompt has number of great features, such as built in help (which you can get to by simpy typing <code>?</code>) or the built in package manager (which can be accessed via <code>]</code>).
</p>
</div>
<br/>

<div style="text-align:center"><img src="/post_pdfs/machine_learning_in_julia/REPL.png" />
</div><br/>

<div style="text-align: justify"><p>
One of the advantages of Julia is that it supports multiple dispatch, a paradigm which means that we never need to know argument types before passing them into a method. This allows us to write generic algorithms which can be easily re-used - a property which makes an open source machine learning framework, like Flux, an enticing prospect. Stefan Karpinski elegantly contrasts this behaviour with single-dispatch, object-orientated languages like C++ in <a href="https://www.youtube.com/watch?v=kc9HwsxE1OY"><i>The Unreasonable Effectiveness of Multiple Dispatch</i></a>.
</p><p>
If you're interested in the Julia language in general, I'd recommend watching <a href="https://www.youtube.com/watch?v=VgZm53qgj9Q">this</a> interview with two of the co-founders, Viral Shah and Jeff Bezanson.
</p></div><br/>

### The magic of Flux
Flux is a fairly young framework with the first commit made in 2016. Consequentially it has been built with modern deep learning architectures in mind. Sequential layers can be chained together with ease, the [Zygote.jl](https://fluxml.ai/Zygote.jl/latest/) dependency takes care of automatic differentiation, and full GPU support is provided by [CUDA.jl](https://juliagpu.gitlab.io/CUDA.jl/), all the while keeping the Flux code-base to a fraction of the size of PyTorch and Tensorflow.

To showcase the framework, we compare two Flux implementations of the customary digit classifier on the MNIST dataset to their Tensorflow and Pytorch equivalents.

#### Functional Comparison: Flux vs Tensorflow
With Tensorflow (TF) being the most widely used deep learning framework in industry, it is worth comparing the Flux API to the Tensorflow functional API. In Flux, we build sequential models by simply chaining together a series of Flux layers. This is demonstrated below where we construct a feed-forward network of two dense layers which follow on from a simple flattening layer. A typical ReLU activation is used in the hidden layer along with dropout for regularisation. This is all neatly wrapped up in a Julia function for us to use in our script.

{% highlight julia %}
function create_model(input_dim, dropout_ratio, hidden_dim, num_classes)
    return Chain(
        Flux.flatten,
        Dense(input_dim, hidden_dim, relu),
        Dropout(dropout_ratio),
        Dense(hidden_dim, num_classes)
    )
end
{% endhighlight %}

Using the functional Keras API, the equivalent TF2 code looks incredibly similar.

{% highlight python %}
def create_model(input_dims, dropout_ratio, hidden_dim, num_classes):
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_dims),
        tf.keras.layers.Dense(hidden_dim, activation='relu'),
        tf.keras.layers.Dropout(dropout_ratio),
        tf.keras.layers.Dense(num_classes)
    ])
{% endhighlight %}
<br/>

Before we start setting up training code, the MNIST data needs to be collated in an easily ingestible way. The `Flux.Data` module has a special `Dataloader` type which handles batching, iteration, and shuffling over the data. Combined with the `onehotbatch` function, this makes generating loaders for the training and test set data pretty straightforward. Notice that in this function, the optional typing of function arguments is showcased.  

{% highlight julia %}
function get_dataloaders(batch_size::Int, shuffle::Bool)
    train_x, train_y = MNIST.traindata(Float32)
    test_x, test_y = MNIST.testdata(Float32)

    train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

    train_loader = DataLoader(train_x, train_y, batchsize=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_x, test_y, batchsize=batch_size, shuffle=shuffle)

    return train_loader, test_loader
end
{% endhighlight %}

Since MNIST is a simple and small dataset, we just use a straightforward implementation for collating the data in TF2.

{% highlight python %}
def get_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test
{% endhighlight %}
<br/>

In our main function we use the two functions defined above to create the dataloaders and create the model. The trainable parameters, which will be passed to the train function, are collected into an object using `Flux.params(model)`. Flux offers a number of optimisers such as `RMSProp`, `ADADelta`, and `AdaMax`, but in this demonstration `ADAM`. Notice that the learning rate is set using the unicode character `η`. This is a really great feature in Julia which makes Flux implementations look closer to the mathematical model as published. The instantiation of the optimiser is followed by the loss function definition, which thanks to the concise Julia syntax, also exhibits a neat mathematical quality. The choice of `logitcrossentropy` (which applies the softmax function to the logit output internally) is commonly used as a more stable alternative to `crossentropy` in classification problems.

All four of the above mentioned components come together in the `Flux.train!` loop, which optimises trainable parameters given the loss function, optimiser, and the training dataloader. The loop is run a number of times using the `@epochs` macro. Notice that the model is contained in the loss function definition and so it does not need to be passed in explicitely. We simply indicate which paramters we want to be optimised. 

{% highlight julia %}
function main(num_epochs, batch_size, shuffle, η)
    train_loader, test_loader = get_dataloaders(batch_size, shuffle)

    model = create_model(28*28, 0.2, 128, 10)
    trainable_params = Flux.params(model)

    optimiser = ADAM(η)
    loss(x,y) = logitcrossentropy(model(x), y)

    @epochs num_epochs Flux.train!(loss, trainable_params, train_loader, optimiser)

    testmode!(model)
    @show accuracy(train_loader, model)
    @show accuracy(test_loader, model)
    println("Complete!")
end
{% endhighlight %}

The TF2 implementation has a similar flow. The data is loaded, model initialised, and the loss function defined. Unlike in the Flux implementation the model is not present in the loss function. Tensorflow builds the computational graph at the point of `compile` on the model and only executes after calling `fit`.

{% highlight python %}
def main(num_epochs, optimiser):
    x_train, y_train, x_test, y_test = get_data()

    model = create_model(
        input_dims=(28, 28),
        hidden_layer_dim=128,
        dropout_ratio=0.2,
        num_classes=10
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=num_epochs)

    print('Evaluate on training data')
    model.evaluate(x_train, y_train, verbose=2)
    print('Evaluate on test data')
    model.evaluate(x_test, y_test, verbose=2)
{% endhighlight %}
