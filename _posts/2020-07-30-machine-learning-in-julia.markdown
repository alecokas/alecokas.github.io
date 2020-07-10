---
layout: post
title:  "Flux: The Flexible Machine Learning Framework for Julia"
date:   2020-07-09 10:00:00 +0000
categories: Julia Flux
---

<div style="text-align: justify">
With Julia having established itself as the 6th most loved language in this year's <a href="https://insights.stackoverflow.com/survey/2020#technology-most-loved-dreaded-and-wanted-languages">Stack Overflow Developer survey</a> and <a href="https://juliacon.org/2020/">JuliaCon 2020</a> kicking off in the next couple weeks, I thought this might be a good time to talk about machine learning in Julia. In this post, I'll touch on Julia and some of the features I enjoy, and then I'll introduce Flux, the pure-Julia machine learning framework.
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

### The magic of Flux.jl
Flux is a fairly young framework, with the first commit made in 2016, that is an entire four years after the historic publication of AlexNet. As a result, the flux community has been able to make decisions with GPU compute and modern machine learning frameworks in mind. 



