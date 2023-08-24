# Malt Transformer

This repository contains an implementation of the GPT architecture in Racket/Malt.

The transfomer built here is trained to produce arithmetic sequences by looking at the digits themselves of a sequence of numbers. E.g. if you see "102 103", then you can infer that the next number should start with "10" and the digit "3" should come after. This is not the best application of a GPT model, but does provide a relatively small baseline task which is relatively easy to reason about.

The transformer defined here only gets around 30% accuracy, but this is significantly better than random (guessing each digit/sepeartor randomly would produce around 9% accuracy), and I believe this is mostly an issue of the size of the model, which was neccesary to get reasonable training times. As malt improves, better performance should be possible.

Racket: [https://racket-lang.org/](https://racket-lang.org/)

Malt: [https://docs.racket-lang.org/malt/index.html](https://docs.racket-lang.org/malt/index.html)

## Repository Structure

The implementation of the architecture itself is in `transformer.rkt`, which contains code for all the non-malt-provided building blocks, as well as an example of training the transformer.

`data/arithmetic-sequences/arithmetic-sequences.rkt` contains the code for generating and correctly formatting training and testing data, in this case arithmetic sequences.

`old/` contains older, out-of-date code as well as experiments and tests which informed decisions of the final version. This folder is safe to ignore. 

## Resources Used

I'm pretty new to this, so the following resources were all very useful for me.

- [https://e2eml.school/transformers.html](https://e2eml.school/transformers.html)
- [https://karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html)
- [https://nlp.seas.harvard.edu/2018/04/03/attention.html](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- The papers under "Resources" in this repo.

And most importantly of course, The Little Learner! [https://www.thelittlelearner.com/](https://www.thelittlelearner.com/) .
