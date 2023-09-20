---
layout: post
title:  "How Self-Attention Works"
permalink: /how-self-attention-works
date:   2023-06-13 19:32:45 -0700
category: "AI"
published: false
---

In this post, I explain the motivation behind scaled dot-product self-attention and how it works.

* What is attention?
* What is self-attention?
* How does it work, at a high level?
* How does it work?

Prerequisite knowledge:
* How neural networks work
* How attention works

What is self-attention?

Self-attention[1] is a mechanism in deep learning for learning “relevance” between elements of a single collection[0]. For example, we could calculate the relevance between each word in the following sentence.
￼
Each pair of words gets a “relatedness” score.
￼

Different types of “related”-ness

Since “relevance” is a learned property, attention mechanisms can learn different semantic meanings of relevance.

Note: In multi-headed self-attention (e.g. in the Transformer architecture), there are multiple attention mechanisms in parallel that can learn different semantic meanings of “relevance” over the same input sequence.

For example, we could have a noun-verb “relevance”.

“The mouse ate some cheese.”

Or we could have a subject-object “relevance”:

“The mouse ate some cheese”.

[ insert graphic ]

How does scaled dot-product self-attention work?

Scaled dot-product self-attention works by computing a “relevance score” between each element of the input collection.

[ insert graphic, illustrating a matrix of relevance scores ]

There are many potential ways of computing “relevance”, but a common way to do this is by using the scaled dot-product.

Self-attention then computes the output of the 

1. Since self-attention operates on vectors, we need to transform each element of the input (e.g. a word) into a vector representation. There are several ways to do this, but it mostly involves using embeddings.
2. Then we need to compute the relevance between the different elements (i.e. vectors).

How do we train scaled dot-product attention?


An example: scaled dot-product attention, step-by-step

For example, we want a layer in a neural net that computes

We start with an input collection, such as a sentence:

“The mouse ate some cheese.”

Then we transform the input collection into a set of vectors.

[ insert graphic showing how we transform the input collection to a set of vectors ]

Firstly, transform the input collection to a set of vectors. For example, if you’re using self-attention on a collection of words, that needs to be transformed into a collection of vectors. One common way of doing this is via a learned embedding.

Now that we have a collection of vectors, we can translate that into a 

[0]: Self-attention operates on unordered collections. However, it can be adapted to be used on collections with implicit structure, like text (i.e. the words have an implicit sequence structure) and images (i.e. the pixels have an implicit grid structure).

[1]: Specifically, scaled dot-product self-attention. There are other less-common algorithms for computing self-attention.

Sources
* Attend, Show, and Tell paper: https://arxiv.org/pdf/1502.03044.pdf
* Attention Is All You Need paper: https://arxiv.org/abs/1706.03762
