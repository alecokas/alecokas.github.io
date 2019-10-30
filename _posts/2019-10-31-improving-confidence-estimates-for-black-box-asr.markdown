---
layout: post
title:  "Improving Confidence Estimates for Black-Box ASR"
date:   2019-10-31 09:00:00 +0100
categories: ASR, Confidence
---

<div style="text-align: justify">
I’ve recently written a paper, <a href="https://arxiv.org/abs/1910.11933"><B>‘Confidence Estimation for Black Box Automatic Speech Recognition Systems using Lattice Recurrent Neural Networks’</B></a>, on how to improve confidence estimates for black-box ASR systems. It is currently under review for <a href="https://2020.ieeeicassp.org/">ICASSP</a>, which is a great conference for anyone interested in deep learning for speech and signal processing. In this post, I’ll touch on some of the topics covered in this work and motivate why confidence estimation for black-box ASR is important. 
</div>
<br/>

### Black-box ASR and why we need it
<div style="text-align: justify">
The intricate nature of speech recognition makes developing an in-house automatic speech recognition (ASR) system challenging (often infeasible) for even medium sized tech companies. Furthermore, the desire to handle a wide array of regional accents, dialects, and languages compounds these difficulties. The most common work-around is to out-source any speech recognition service to dedicated providers such as <a href="https://www.speechmatics.com/">Speechmatics</a>, or large tech companies with a dedicated ASR team (such as Amazon, Google, Apple, etc.). These services are usually assessible through an <a href="https://en.wikipedia.org/wiki/Application_programming_interface">API</a> call to the procured cloud-based service which returns a sequence of words as a predicted transcription.
</div>
<br/>

### So, where’s the problem?
<div style="text-align: justify">
This model sounds incredibly convenient right? Well, it is - until something breaks. At which point you wonder how reliable the predicted transcriptions are. Since the predictions are presented via an API, the end user doesn’t have access to the internal state of the ASR system. In a <a href="https://alecokas.github.io/asr/2019/09/30/lattices-for-asr.html">previous post</a>, I presented a few common structures available at the decoding stage of the ASR output: one-best word sequences, confusion networks, and lattices. The API will often simply present the one-best word sequence without any additional information even though it is cheaply available. In our paper, my co-authors and I show that by even adding simple features such as word durations and word posteriors, a reasonable estimate of confidence can be obtained using a Lattice Recurrent Neural Network. The nature of this model allows it to operate on lattices, thereby obtaining confusion information from competing arcs, and easily incorporate additional features which are simple to obtain from most black-box ASR systems. We similarly explore the use of an encoder for representing sub-word level features in a fixed form such that <a href="http://mi.eng.cam.ac.uk/foswiki/pub/Main/KK492/graphemic-lexicons-spoken.pdf">grapheme features</a> can be introduced into the model.
</div>
<br/>

### Ask for more!
<div style="text-align: justify">
In short, we argue that industry professionals should insist on being given the decoded lattices or confusion networks rather than just the restrictive one-best word sequence. Furthermore, cloud-based ASR systems are often able to provide simple features which already exist within the black-box system if requested. These features such as the word duration, mapped word posterior, and grapheme features have significance when attempting to determine the reliability of a candidate transcription.
</div>
<br/>

### TL;DR
<div style="text-align: justify">
We wrote a paper which argues for obtaining richer features and competing hypothesis information from black-box ASR systems in order to improve confidence scores.
</div>
<br/>
