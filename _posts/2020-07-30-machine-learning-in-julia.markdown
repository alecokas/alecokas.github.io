---
layout: post
title:  "Flux: The Flexible Machine Learning Framework for Julia"
date:   2020-07-09 10:00:00 +0000
categories: Julia Flux
---

<div style="text-align: justify">
With Julia having established itself as the 6th most loved language in this year's <a href="https://insights.stackoverflow.com/survey/2020#technology-most-loved-dreaded-and-wanted-languages">Stack Overflow Developer survey</a> and <a href="https://juliacon.org/2020/">JuliaCon 2020</a> kicking off in the next couple weeks, I thought this might be a good time to talk about machine learning in Julia. So in this post, I'll touch on Julia and some of the features I enjoy, and then I'll introduce Flux, the pure-Julia machine learning framework.
</div>
<br/>


### The Julia Language
<div style="text-align: justify"><p>
The founders of Julia were looking to create a language which was geared towards interactive scientific computing, whilst at the same time supporting more advanced software engineering via JIT compilation to native code. To facilitate this dual purpose, Julia is dyamically typed with the default behaviour being to allow values of any type when no type specification is provided. Additional expressiveness is permissible via optional typing which provides some of the efficiency gains typically associated with static languages. Having spent my formative programming years writing C++ code, and now almost exclusively working in Python, the option to type my machine learning code is incredibly appealing.
</p>
<p>
The REPL interactive prompt has number of great features, such as built in help (which you can get to by simpy typing <code>?</code>) or the built in package manager (which can be accessed via <code>]</code>).
</p>
</div>
<br/>

<div style="text-align:center"><img src="/post_pdfs/machine_learning_in_julia/REPL.png" />
</div><br/>

### The magic of Flux.jl

Julia is a fairly young language, with the founders starting development in 2009, just three years before the historic publication of AlexNet. As a result, the community has been able to make decision which played into  



