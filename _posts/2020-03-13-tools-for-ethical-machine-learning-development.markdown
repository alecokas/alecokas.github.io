---
layout: post
title:  "Tools for Ethical Machine Learning Development"
date:   2020-03-01 09:00:00 +0100
categories: Ethics
---

<div style="text-align: justify">
Making use of simple tools to reinforce the importance of ethical development in machine learning can go a long way to ensuring that our solutions meet a standard of fairness and are ethically sound. This is particularly important when operating under time constraints or when putting forward a new explorative project. In this post, I'm going to (very briefly) advocate two simple ideas for assisting a more ethically conscious workflow: <em>ethics checklists</em> and <em>model audits</em>.
</div>
<br/>

### Ethics checklists
<div style="text-align: justify">
A checklist of ethical questions can act as a concrete reference point, or simply a central motif, from which to spark actionable debate about how to move forward with new or existing machine learning models. Recently, I came across <a href="https://deon.drivendata.org//">Deon</a>, an open source Python tool which can be used to add an ethics checklist to a machine learning project. The idea behind this is that we should maintain an <code>ETHICS.md</code> document containing an explicit checklist of ethical considerations and precautions taken during the development of our projects. Examples of items in the checklist include: user privacy, data biases, proxy discrimination. This mirrors the <code>README.md</code> type appraoch to documentation, which I believe strikes a good balance between simplicity and utility. 
</div>
<br/>
<div style="text-align: justify">
The Deon command line tool makes this actionable by providing a simple interface to generate and customise such an ethics checklist. To use their default checklist (which is a great starting point), simply run the following:

<pre>$ pip install deon
$ deon -o ETHICS.md</pre>

The default checklist groups 20 items into 5 sections:
<ol>
  <li>Data Collection</li>
  <li>Data Storage</li>
  <li>Analysis</li>
  <li>Modeling</li>
  <li>Deployment</li>
</ol>  
The relative importance of each of these sections will vary depending on the specifics of your project. For instance, one can imagine that an industry application may require a higher standard for data storage and deploment items than a internal academic project. Take a look at the Deon <a href="https://github.com/drivendataorg/deon/">repository</a> and <a href="https://github.com/drivendataorg/deon/wiki/Overview">wiki</a> for more information on how to use their tool how ethics checklists can support your machine learning project.
</div>
<br/>

### Model audits
<div style="text-align: justify">
Once an ethics checklist has been set up, one way to assess the predictions being made by the model is by an <em>audit</em>. This involves a programmatic assessment of the model predictions as well as the ground truth labels with the aim of uncovering biases or prejudice. From the obvious applications, such as medical and autonomous vehicles, through to legal and financial recommendations, the importance of ensuring fair models is essential. The Center for Data Science and Public Policy at the University of Chicago have developed a bias and fairness audit tool called <a href="http://www.datasciencepublicpolicy.org/projects/aequitas/">Aequitas</a>. Aequitas can be used as a Python library, a command line tool, or via a web interface to generate a report as well as detailed statistics about bias and fairness in your model. This assessment is done for a number of bias metrics which are defined in the <em>fairness tree</em>:
</div>

<img src="http://www.datasciencepublicpolicy.org/wp-content/uploads/2020/02/Fairness-Weeds.png">

<div style="text-align: justify">
From left to right, the parities depicted in the bottom row of the fairness tree are:
<ul>
  <li>False Positive</li>
  <li>False Discovery Rate</li>
  <li>False Positive Rate</li>
  <li>Recall</li>
  <li>False Negative</li>
  <li>False Omission Rate</li>
  <li>False Negative Rate</li>
</ul>
Although some guidelines are provided in the fairness tree, determining which metric is most critical depends on the particular circumstances of your application. I believe that audit tools, such as Aequitas, together carefully selected metrics provide a starting point from which evaluating and benchmarking unfair bias within a model can be documented and iteratively reduced. To find out more about the details of Aequitas read the paper references below.
</div>
<br/>


#### References
Saleiro, Pedro, et al. "Aequitas: A bias and fairness audit toolkit." arXiv preprint arXiv:1811.05577 (2018).
