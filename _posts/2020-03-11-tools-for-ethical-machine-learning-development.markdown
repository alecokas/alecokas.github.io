---
layout: post
title:  "Tools for Ethical Machine Learning Development"
date:   2020-03-11 09:00:00 +0000
categories: Ethics
---

<div style="text-align: justify">
Making use of simple tools to reinforce the importance of ethical development in machine learning can go a long way to ensuring that our solutions meet a standard of fairness and are morally sound. This is particularly important when operating under time constraints or when putting forward a new explorative project. In this post, I'm going to (very briefly) advocate two simple ideas for assisting a more ethically conscious workflow: <em>ethics checklists</em> and <em>model audits</em>.
</div>
<br/>

### Ethics checklists
<div style="text-align: justify">
A checklist of ethical questions can act as a concrete reference point, or simply a central motif, from which to spark actionable debate about how to move forward with new or existing machine learning models. Recently, I came across <a href="https://deon.drivendata.org//">Deon</a>, an open source Python tool which can be used to add an ethics checklist to a machine learning project. The idea behind this is that we should maintain an <code>ETHICS.md</code> document containing an explicit checklist of ethical considerations and precautions taken during the development of our projects. Examples of items in the checklist include: user privacy, data biases, and proxy discrimination. This mirrors the <code>README.md</code> type appraoch to documentation, which I believe strikes a good balance between simplicity and utility. 
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
The relative importance of each of these sections will vary depending on the specifics of your project. Furthermore, one can imagine that an industry application may be required to uphold a higher standard for the data storage and deploment items than perhaps a internal academic project. An example of an issue which could have benefitted from an ethical checklist is <a href="http://content.time.com/time/business/article/0,8599,1954643,00.html">Nikon's facial recognition</a> system detecting Asian eyes as being closed. The collection bias item in Deon's default checklist may have raised enough questions for an appropriate dataset which includes Asian faces to be curated. Take a look at the Deon <a href="https://github.com/drivendataorg/deon/">repository</a> and <a href="https://github.com/drivendataorg/deon/wiki/Overview">wiki</a> for more information on how to use their tool, and how an ethics checklist can support your machine learning project.
</div>
<br/>

### Model audits
<div style="text-align: justify">
Once an ethics checklist has been set up, one way to assess the fairness of the predictions being made by the model is by an <em>audit</em>. This involves a programmatic assessment of the model predictions as well as the ground truth labels with the aim of uncovering biases or prejudice. From obvious applications, such as medical and autonomous vehicles, through to legal and financial recommendations, the importance of ensuring the fairness of our models is essential. To this end, the <em>Center for Data Science and Public Policy</em> at the <em>University of Chicago</em> has developed a bias and fairness audit tool called <a href="http://www.datasciencepublicpolicy.org/projects/aequitas/">Aequitas</a>. Aequitas can be used as a Python library, a command line tool, or via a web interface to generate a report as well as detailed statistics about biases and the fairness in your model. This assessment is done for a number of bias metrics, defined in the <em>fairness tree</em>.
</div>

<div style="text-align:left"><img src="/post_pdfs/tools_for_ethical_machine_learning_development/Fairness-Weeds-1200x897.png" />
</div><br/>

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
Although some guidelines are provided in the fairness tree, determining which metric is most critical depends on the particular circumstances of your application.

In high risk applications, such as the <a href="https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis">COMPAS Recidivism Algorithm</a> (which has been shown to be racially biased), assessing group biases using distributional and group-based error metrics can expose flaws in the fairness of a model before it goes into production. This provides the opportunity for preemptive correctional measures to be introduced, or the termination of a potentially harmful project. Audit tools, such as Aequitas, together with carefully selected metrics provide a starting point from which evaluating and benchmarking unfair bias within a model can be documented and iteratively reduced. Such an evaluation should not be carried out exclusively by machine learning practitioners. End-users and policymakers have a responcibility to carry out audits of their own, particularly in high stakes applications as mentioned above. To find out more about the details of Aequitas, read the paper in the references below and have a look at the <a href="https://github.com/dssg/aequitas">repository</a>.
</div>
<br/>

#### References
Saleiro, Pedro, et al. "Aequitas: A bias and fairness audit toolkit." arXiv preprint arXiv:1811.05577 (2018).
