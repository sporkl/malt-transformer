#lang racket

(require malt)

; TODO: looks like batch dim and heads dim are swapped before attention is calculated
; might not be an issue, might actually work around potential problem of not concatenating along right dimension

; TODO: figure out why flat-tensors is getting in invalid form (where the shape in the representation is a tensor for some reason)
; it's the same thing for both flat-tensors and nested-tensors: a dual is being passed to a function that can't handle duals
; issue is that list->tensor expects list of non-dual tensors

; n = # of words inputted to the model
; N = vocabulary size/length of one-hot-vectors which encode words
; d_model = dimension of embedding space for the model in general (outputted from each attention block)
; d_k = dimension of embedding space for queries and keys
; d_v = dimension of embedding space for values
; h = # heads per multi-head-attention block
; x, y, z = misc. variables

; LAYER NORMALIZATION

; sum-matrix
; (list x y) -> (list)
(define sum-matrix
  (lambda (t)
    (sum (sum t))))

; mean-matrix
; (list n d_model) -> (list)
(define mean-matrix
  (lambda (m n)
    (lambda (t)
      (/ (sum-matrix t) (* m n)))))

; standard deviation of matrix entries
; (list n d_model) -> (list)
(define stddev-matrix
  (lambda (m n)
    (lambda (t)
      (let ([m ((mean-matrix m n) t)])
        (sqrt ((mean-matrix m n) (expt (- t m) 2)))))))

; normalization
; (list n d_model) -> (list n d_model)
(define normalize
  (lambda (m n)
    (lambda (t)
      (let ([m ((mean-matrix m n) t)]
            [σ ((stddev-matrix m n) t)])
        (/ (- t m) σ)))))

; normalization as a layer function
; this includes shift and scale after, as in the paper
(define normalize-layer
  (lambda (m n)
    (lambda (t)
      (lambda (θ)
        (let ([n ((normalize m n) t)])
          ((linear n) θ))))))

; CONCATENATION

; (list h n dv) -> (list n (* h dv))

#| (define tensor-drop vector-drop) |#
#| (define tensor-take vector-take) |#
#| (define tensor-append vector-append) |#

#| ; append-vectors-1-1 |#
#| ; (list x) (list y) -> (list (+ x y)) |#
#| (define append-vectors-1-1 |#
#|   (prim2 |#
#|     (lambda (u v) |#
#|       (tensor-append u v)) |#
#|     (lambda (ra rb z) |#
#|         (values |#
#|           (tensor-take z (tlen ra)) |#
#|           (tensor-drop z (tlen ra)))))) |#

#| (define append-vectors |#
#|   (ext2 append-vectors-1-1 1 1)) |#

; takes a tensor of tensors
; and concatenates them along the vectors
; TODO: verify that works correctly even when there is a batch dimension
(define concat-vectors
  (lambda (t)
    (concat-vectors-helper t 1 (tref t 0))))

(define concat-vectors-helper
  (lambda (t i a)
    (cond
      [(eqv? i (tlen t)) a]
      [else
        (concat-vectors-helper t (add1 i) (concat a (tref t i)))])))

; ATTENTION

; softmax not as a layer function
; (list x) -> (list x)
(define softmax-f
  (lambda (t)
    (let ((z (- t (max t))))
      (let ((expz (exp z)))
        (/ expz (sum expz))))))

#| (define softmax-f |#
     #|   (ext1 softmax-f-1 1)) |#

#| ; attention |#
#| ; (list n dk) (list n dk) (list n dv) -> (list n dv) |#
#| ; works with vector q as well, easier to conceptualize: (list dk) (list n dk) (list n dv) -> (list dv) |#
#| (define attention |#
     #|   (lambda (Q K V) |#
            #|     (sum-cols (*-2-1 V |#
                                    #|         (softmax (/ |#
                                                           #|                    (dot-product-2-1 K Q) |#
                                                           #|                    (sqrt (vlen Q)))))))) |#

; for some reason the malt-provided version doesn't seem to work. Seems to be an ext2 1 1 for some reason, not sure why
(define *-2-1 (ext2 * 2 1))

; attention
; (list n d_k) (list n d_k) (list n d_v) -> (list n d_v)
; should work with vector q as well, easier to conceptualize: (list d_k) (list n d_k) (list n d_v) -> (list d_v)
; d_k is treated as a hyperparameter so it works nicely with automatic differentiation
(define attention
  (lambda (d_k)
    (lambda (Q K V)
      (let* ([scores (dot-product-2-1 K Q)]
             [processed-scores
               (softmax-f (/ scores (sqrt d_k)))]
             [vals (*-2-1 V processed-scores)])
        (sum-cols vals)))))

; create masking matrix
(define make-future-mask
  (lambda (n)
    (build-tensor
      (list n n)
      (lambda (p)
        (let ([x (ref p 1)]
              [y (ref p 0)])
          (cond
            [(> x y) -inf.0]
            [else 1]))))))

; attention with masking
; (list n d_k) (list n d_k) (list n d_v) -> (list n d_v)
(define masked-attention
  (lambda (n d_k)
    (lambda (Q K V)
      (let* ([scores (dot-product-2-1 K Q)]
             [masked-scores
               (* scores
                  (make-future-mask n))]
             [processed-scores
               (softmax-f
                 (/ masked-scores (sqrt d_k)))]
             [vals (*-2-1 V processed-scores)])
        (sum-cols vals)))))

; attention layer function
(define attention-layer
  (lambda (d_k)
    (lambda (t)
      (lambda (θ)
        (let
          ([Q ((linear t) θ)]
           [K ((linear t) (refr θ 2))]
           [V ((linear t) (refr θ 4))])
          ((attention d_k) Q K V))))))

; masked attention layer function
(define masked-attention-layer
  (lambda (n d_k)
    (lambda (t)
      (lambda (θ)
        (let
          ([Q ((linear t) θ)]
           [K ((linear t) (refr θ 2))]
           [V ((linear t) (refr θ 4))])
          ((masked-attention n d_k) Q K V))))))

; PARALLEL BLOCK
; runs the same block h times in parallel
; concatenation handled seperately

; given block b and parallization h
; if b takes a θ of the form (list (list a...) (list b...))
; then parallel-block θ should be of the form (list (list h a...) (list h b...))
; so that each invocation of b's block-fn is vectorized h times

; block fn should duplicate t b times

; in the context of attention

; parallelize-shape-list
; takes a shape list and prepends each shape with a number h
; this should make a function called with this list be repeated h times
(define parallelize-shape-list
  (lambda (s h)
    (cond
      [(null? s) '()]
      [else
        (cons
          (cons h (car s))
          (parallelize-shape-list (cdr s) h))])))

; nlicate-tensor
; h and tensor with (shape t) -> tensor with shape (cons h (shape t))
(define nlicate-tensor
  (lambda (n)
    (lambda (t)
      (let* ([st (shape t)]
             [wt (reshape (cons 1 st) t)])
      (nlicate-tensor-helper
        (sub1 n) wt (concat-n (rank wt)) wt)))))

(define nlicate-tensor-helper
  (lambda (n t cf a)
    (cond
      [(eqv? n 0) a]
      [else
        (nlicate-tensor-helper
          (sub1 n) t cf (cf t a))])))

; parallel-block
; takes in a block and a number h
; and returns a tensor of size h,
; where each member of that tensor is a different result of running the block on the input
(define parallel-block
  (lambda (b h)
    (block
      (lambda (t)
        (lambda (θ)
          (((block-fn b) ((nlicate-tensor h) t)) θ)))
      (parallelize-shape-list
        (block-ls b)
        h))))

#| (define parallel-block |#
#|   (lambda (b h) |#
#|     (block |#
#|       (block-fn b) |#
#|       (parallelize-shape-list (block-ls b) h)))) |#

; concat-vectors-block
; block takes in a tensor of (shape h n dv)
; returns a tensor of shape (n (* h dv))
(define concat-vectors-block
  (lambda ()
    (block
      (lambda (t)
        (lambda (θ)
          (concat-vectors t)))
      (list))))

; BLOCKS

; skip block
(define skip-block
  (lambda (b)
    (block
      (lambda (t)
        (lambda (theta)
          (+ t (((block-fn b) t) theta))))
      (block-ls b))))

; positional encoding block for learned positional encoding
; (list n d_model) -> (list n d_model)
(define positional-encoding-block
  (lambda (n d_model)
    (block
      (lambda (t)
        (lambda (θ)
          (+ t (ref θ 0))))
      (list (list n d_model)))))

; layer normalization block
; (list n d_model) -> (list n d_model)
(define normalize-block
  (lambda (n d_model)
    (block
      (normalize-layer n d_model)
      (list
        (list d_model d_model)
        (list d_model)))))

; relu/dense block
(define dense-block
  (lambda (n m)
    (block relu
      (list
        (list m n)
        (list m)))))

; linear block
; like a relu/dense block but without the rectification
(define linear-block
  (lambda (n m)
    (block
      (lambda (t)
        (lambda (theta)
          ((linear t) theta)))
      (list
        (list m n)
        (list m)))))

; might need to swap d_model and d_k

; single attention block
(define attention-block
  (lambda (d_model d_k d_v)
    (block
      (attention-layer d_k)
      (list
        (list d_k d_model) ; linear for Q
        (list d_k)
        (list d_k d_model) ; linear for K
        (list d_k)
        (list d_v d_model) ; linear for V
        (list d_v)))))

; single masked attention block
(define masked-attention-block
  (lambda (n d_model d_k d_v)
    (block
      (masked-attention-layer n d_k)
      (list
        (list d_k d_model) ; linear for Q
        (list d_k)
        (list d_k d_model) ; linear for K
        (list d_k)
        (list d_v d_model) ; linear for V
        (list d_v)))))

; mutli-head attention block
; (list n d_model) -> (list n d_model)
(define multi-head-attention-block
  (lambda (d_model d_k d_v h)
    (stack-blocks
      (list 
        (parallel-block
          (attention-block d_model d_k d_v)
          h)
        (concat-vectors-block)
        (linear-block d_model (*-ρ h d_v))))))

; maked multi-head attention block
; (list n d_model) -> (list n d_model)
(define masked-multi-head-attention-block
  (lambda (n d_model d_k d_v h)
    (stack-blocks
      (list
        (parallel-block
          (masked-attention-block n d_model d_k d_v)
          h)
        (concat-vectors-block)
        (linear-block d_model (*-ρ h d_v))))))

; feedforward block
; this is the second part of the transformer block
(define feedforward-block
  (lambda (d_model)
    (stack-blocks
      (list
        (dense-block d_model d_model)
        (linear-block d_model d_model)))))

; transformer block
; repeat this 12 times to make gpt-3!
(define transformer-block
  (lambda (n d_model d_k d_v h)
    (stack-blocks
      (list
        (skip-block
          (masked-multi-head-attention-block n d_model d_k d_v h))
        (normalize-block n d_model)
        (skip-block
          (feedforward-block d_model))
        (normalize-block n d_model)))))

; softmax block
; (list n N) -> (list n N)
(define softmax-block
  (lambda ()
    (block softmax (list))))
; malt-included softmax here, NOT softmax-f

; TRANSFORMER

; here it is!
; n = 15
; N = 11
; d_model = 8
; d_k = 2
; d_v = 2
; h = 4
; batch size 1 (increase when get things working)
; repeat 3 times
(define counter-transformer-network
  (stack-blocks
    (list
      (linear-block 11 8)
      (positional-encoding-block 15 8)
      (transformer-block 15 8 2 2 4)
      (transformer-block 15 8 2 2 4)
      (transformer-block 15 8 2 2 4)
      (linear-block 8 11)
      (softmax-block))))

; get the data
(require "data/arithmetic-sequences/arithmetic-sequences.rkt")

; train and test funcs

(define train-counter
  (λ (network)
    (with-hypers ; TODO: use grid search
      ((alpha 0.0005)
       (revs 1) ; change to 20000
       (batch-size 1)
       (mu 0.9)
       (beta 0.999))
      (trained-transformer (block-fn network) (block-ls network)))))

(define trained-transformer
  (λ (transformer theta-shapes)
    (model transformer
      (adam-gradient-descent
        (sampling-obj
          ((with-recording l2-loss)
            transformer)
           sequences-train-xs sequences-train-ys)
        (init-theta theta-shapes)))))

(define train-and-test-transformer
  (λ (network)
    (fprintf (current-error-port) "Accuracy: ~a~%"
      (accuracy
        (train-counter network)
        sequences-test-xs sequences-train-ys))))

; train!

(define product (lambda (l) (foldl (lambda (x y) (* x y)) 1 l)))
(define sum-l (lambda (l) (foldl (lambda (x y) (+ x y)) 0 l)))
(sum-l (map product (block-ls counter-transformer-network))) ; with current params should be half the size of morse-fcn

(start-logging)
(train-and-test-transformer counter-transformer-network)
