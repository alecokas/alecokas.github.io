---
layout: post
title:  "Tools for Ethical Machine Learning Development"
date:   2020-03-13 09:00:00 +0100
categories: Ethics
---

<div style="text-align: justify">
Making use of simple tools to reinforce the importance of ethical development in machine learning can go a long way to ensuring that our solutions meet a standard of fairness and are ethically sound. This is particularly important when operating under time constraints or when putting forward a new explorative project. In this post, I'm going to (very briefly) advocate two ideas simple ideas for assisting a more ethically conscious workflow: <em>ethics checklists</em> and <em>model audits</em>.
</div>
<br/>

### Ethics checklists
<div style="text-align: justify">
A checklist of ethical questions can act as a concrete reference point or simply a central motif to spark actionable debate when considering how to procede with new or existing machine learning models. Recently, I came across <a href="https://deon.drivendata.org//">Deon</a>, an open source Python tool which can be used to simply add an ethics checklist to new or existing machine learning projects. The idea behind Deon is that just like we expect a <code>README.md</code> file to provide the starting point for a project's documentation, we should expect an <code>ETHICS.md</code> file containing an explicit checklist of ethical considerations which dictates how the machine learning or data science project is developed. The Deon command line tool makes this actionable by providing a simple interface to generate and customise such an ethics checklist. To use their default checklist (which is a great starting point), simply run the following:

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

### Auditing a model
<div style="text-align: justify">
</div>
<br/>

### TL;DR
<div style="text-align: justify">
We wrote a paper which argues for obtaining richer features and competing hypothesis information from black-box ASR systems in order to improve confidence scores.
</div>
<br/>
