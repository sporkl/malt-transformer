# Malt Transformer

*NOTE:* this implementation is in-progress, and not yet complete.

An implementation of the transformer in racket/malt.

The goal here is to create a transformer that can continue arithmetic sequences of numbers, e.g. 1 2 3, or 47, 50, 53, 56, etc.
The vocabulary has 11 characters: "0123456789 .". Maybe also something to denote start and end? Probably neccesary, using "."

There might be a better goal that's better-fitted for the transformer architecture. I was trying to think of one that had a relatively small vocabulary and was a relatively easy task so that it would be feasible to do with a smaller network.

Roughly speaking transformer architecture looks like this:

input embedding -> positional encoding -> (masked multi-head attention layer -> layer normalization -> skip connection -> relu layer -> layer normalization) repeated -> linear layer -> softmax layer

need to implement (not provided with malt as is):
- positional encoding (can be learned though, so might not be an issue) (implemented to be learned)
- single-head masked attention (implemented)
	- masking (should be easy, just tensor multiplying by specifically-crafted tensor) (implemented)
	- scaling (should be easy, is just tensorized division) (implemented)
- multi-head attention (should be easy once have single-head attention) (implemented)
- concatenation
- layer normalization (implemented)

then can just stack-blocks em up

Things to figure out before implementing:
- can matrix multiplications as in the attention paper be done as vectorized dot products, so easier to do in malt? (yes)
- linear layers are not technically "artificial neurons" in that they are not rectified, correct? (yes)
- what are the parameters learned in the attention layer? is it Q, K, V, or are those just inputs? (linear layers before attention)
- how to properly do embeddings? just linear layer? (yep)

notes:
- Q, K, V are determined by multiplying input vector (embedded) by a learned weight matrix for each. WEIGHT MATRIX for Q, K, V, IS APPLIED BEFORE ATTENTION!
- illustrated transformer explains attention with vectors very clearly, probably easiest to refer to that for implementation
- concatenation step gets multiplied by another weight matrix, Wo, before being sent forward. IMPORTANT FOR CONCAT LAYER. Is this just linear layer? Yep, is just linear layer after concat. So seperate from concat
- the purpose of linear layers in general seems to be to navegate around different dimensionalities
- softmax is already implemented
- I think the weight matrix layers for Q, K, V are the linear layers that come before the scaled dot product attention in the multi-head attention diagram in the og paper
- scaled dot product attention: "compute the dot product of the query with all the keys"
- within an encoder/decoder block, the same feedforward neural network is applied to each word. can be thought of as convolution? No, more along the lines of just vectorized linear I think
- first attention dot product is current query with every key to get score (then scale and softmax). second multiplication is each value vector multiplied each processed score. then summed.
- for ext to work, function needs to be a primitive with a ρ and ∇. so won't work for transposition or vlen by default.
- might be able to get around vlen by having hyperparameter argument to attention.
- should probably implement transposition/swap-dims as a primitive
- what is derivative of concatenation or flattening?
- learned positional encoding can just be addition I think. positional encoding in paper is just addition of a fancy term.
- embedding layer is just linear layer with no bias
- I think embedding layer and positional encoding layer could be combined into linear layer, but is more clear to seperate them. Also, then can use different positional encoding.
- feedforward layer in paper is relu followed by linear
- can use ext1-ρ to extend functions which do not have a gradient, e.g. tlen, reshape, etc.

2023-06-10 notes:
- still stuck on tensor concatenation. getting contract violation for \*: expected: number? given: 4 . kind of bizzare.
- does my prim version of reshape support differentiation for nested and flat tensors? yes it does!

*Resources:*

https://e2eml.school/transformers.html and all resources linked at botton
- Attention is All You Need paper
- Layer Normalization paper
- GPT-3 paper perhaps? need to cite something for getting rid of encoder side.
