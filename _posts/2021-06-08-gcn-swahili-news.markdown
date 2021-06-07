---
layout: post
title:  "Graph Neural Networks for Swahili News Classification"
date:   2021-06-08 08:00:00 +0100
categories: swahili GNN
---
Over the winter, I (along with my collaborator [Tyler Martin](https://www.linkedin.com/in/tyler-martin12/)) have been looking into Graph Neural Networks for Swahili news categorisation. The findings of this work were accepted into the [AfricaNLP 2021](https://sites.google.com/view/africanlp-workshop/home) workshop. Subsequently, an extended abstract version of this work was accepted at [SIGTYP 2021](https://sigtyp.io/ws2021.html) and a summary published in their [April 2021 newsletter](https://sigtyp.io/sigtyp-newsletter-Apr-2021.html).

For this post, I'll simply quote the summary and encourage you to read the details of our work as [published in AfricaNLP 2021](https://arxiv.org/abs/2103.09325) or the [extended abstract in SIGTYP 2021](https://sigtyp.io/workshops/2021/abstracts/13.pdf).


### Graph Convolutional Network for Swahili News Classification
> African languages are underrepresented in both the academic field of natural language processing (NLP) as well as the industry setting. This leads to a shortage of annotated datasets, purpose-built software tools, and limited literature comparing the efficacy of techniques developed for high-resource languages in a low-resource context. Swahili, being the most widely spoken African language, is no exception to this trend.
>
> Our work attempts to address this disparity by providing a set of accessible traditional NLP benchmarks for the task of semi-supervised Swahili news classification. These baseline models include TF-IDF representations, pre-trained fastText embeddings, and Distributed Bag of Words representations each followed by a logistic regression layer. We draw particular attention to the semi-supervised context as this most accurately exemplifies the annotation constraints facing many Swahili text classification tasks.
>
> Graph Neural Networks, a family of model architectures that can leverage implicit inter-document and intra-document relationships, has demonstrated remarkable performance in semi-supervised text classification tasks. As a result, we apply a Text Graph Convolution Network (Text GCN) to our news classification task. To our knowledge, this is the first time a Graph Neural Network has been for text classification for any African language. The experiments demonstrate that Text GCN outperforms the previously described baselines, especially when the proportion of labelled training set documents is reduced below 5% of the full training set. Furthermore, we present a Text GCN variant that is successfully able to maintain similar predictive performance whilst reducing the memory footprint and cloud cost of the model by factors of 5 and 20 respectively. This is achieved by replacing the naive one-hot representation of the nodes in the Text GCN graph with an appropriate bag of words representation.
>
>The empirical results demonstrate the ability of graph-based models to outperform traditional baseline models for this task. We hope that the experimental results and freely available code contribute to addressing the shortage of accessible resources for semi-supervised text classification in Swahili. 


If you found this interesting, definitely [sign up to SIGTYP 2021](https://sigtyp.io/ws2021-schedule.html) (it's free!) where you'll be able to chat to us and take a look at all the other amazing research as well.