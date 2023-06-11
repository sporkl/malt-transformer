#lang racket

(require malt)

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
        (sqrt (mean-matrix (expt (- t m) 2)))))))

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

; TODO: figure out concatenation

#| ; append-vectors |#
#| ; (list x) (list y) -> (list (+ x y)) |#
#| (define append-vectors-1-1 |#
#|   (lambda (u v) |#
#|     (let ([lu (tlen u)] |#
#|           [lv (tlen v)]) |#
#|       (reshape |#
#|         (list (+ lu lv)) |#
#|         (list->tensor (list u v)))))) |#
#| [[1 2 3 4 5 11 12 13 14 15] [6 7 8 9 10 16 17 18 19 20]] |#

; swap the items at position i and j in list l
(define list-swap
  (lambda (l i j)
    (let ([x (list-ref l i)]
          [y (list-ref l j)])
      (list-set
        (list-set l i y)
        j x))))

(define list-remove-index
  (lambda (l i)
    (cond
      [(eqv? i 0) (cdr l)]
      [else (cons (car l) (list-remove-index (cdr l) (sub1 i)))])))

(define list-set
  (lambda (l i v)
    (cond
      [(eqv? i 0) (cons v (cdr l))]
      [else (cons (car l) (list-set (cdr l) (sub1 i) v))])))

; takes a tensor list of indices, and goes to the corresponding index.
(define rtref
  (lambda (t is)
    (cond
      [(null? is) t]
      [else (rtref (tref t (car is)) (cdr is))])))

; swaps dimensions at i and j in a tensor
; e.g. (list x y) 0 1 -> (list y x)
; kind of a generalized transpose
; I believe it is it's own derivative wrt t as well
(define swap-dims-base
  (lambda (t i j)
    (build-tensor
      (list-swap (shape t) i j)
      (lambda (p)
        (rtref t (list-swap p i j))))))

; I believe this is the differentiable version of swap-dims
(define swap-dims
  (lambda (i j)
    (prim1
      (lambda (t) (swap-dims-base t i j))
      (lambda (ra z)
        (swap-dims-base z i j)))))

; I believe this is a differentiable version of reshape
; does using this reshape get things working for flat and nested tensors? looks like it I think
(define d-reshape
  (lambda (s)
    (prim1
      (lambda (t) (reshape s t))
      (lambda (ra z)
        (reshape (shape ra) z)))))

; move-dims
; moves dimension i to dimension j where i <= j
; preserving order of other dimensions
(define move-dims
  (lambda (i j)
    (lambda (t)
      (cond
        [(eqv? i j) t]
        [else
          ((move-dims (add1 i) j)
            (swap-dims i (add1 i) t))]))))

; concat-shape
; returns the appropriate shape for after a concatenation
; to be passed to the reshape within concat
(define concat-shape
  (lambda (c d)
    (lambda (s)
      (list-remove-index
        (list-set s d (* (ref s c) (ref s d)))
        c))))

; concat
(define concat
  (lambda (c d)
    (lambda (t)
      ((d-reshape ((concat-shape c d) (shape t)))
       ((move-dims c (sub1 d)) t)))))

((concat 0 1) (tensor (tensor 1 2) (tensor 3 4) (tensor 5 6)))

; ATTENTION

; softmax not as a layer function
; (list x) -> (list x)
(define softmax-f-1
  (lambda (t)
    (let ((z (- t (max t))))
      (let ((expz (exp z)))
        (/ expz (sum expz))))))

(define softmax-f
  (ext1 softmax-f-1 1))

#| ; attention |#
#| ; (list n dk) (list n dk) (list n dv) -> (list n dv) |#
#| ; works with vector q as well, easier to conceptualize: (list dk) (list n dk) (list n dv) -> (list dv) |#
#| (define attention |#
#|   (lambda (Q K V) |#
#|     (sum-cols (*-2-1 V |#
#|         (softmax (/ |#
#|                    (dot-product-2-1 K Q) |#
#|                    (sqrt (vlen Q)))))))) |#

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

; parallel-block
; takes in a block and a number h
; and returns a tensor of size h,
; where each member of that tensor is a different result of running the block on the input
(define parallel-block
  (lambda (b h)
    (block
      (block-fn b)
      (parallelize-shape-list
        (block-ls b)
        h))))

; BLOCKS
; TODO: fill out blocks

; embedding block
; (list n N) -> (list n d_model)
(define embedding-block
  (lambda (N d_model)
    (block
      (lambda (t)
        (lambda (θ)
          (linear
            (ref θ 0)
            (zero-tensor (list d_model))))
      (list (list d_model N))))))

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
    (block linear
      (list
        (list m n)
        (list m)))))

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
; TODO (requires concatenation)

; feedforward block
; this is the second part of the transformer block
(define feedforward-block
  (lambda (d_model)
    (stack-blocks
      (list
        (dense-block d_model d_model)
        (linear-block d_model d_model)))))

; transformer block
; (list n d_model) -> (list n d_model)
; this is the thing repeated multiple times in the paper, with the grey fill
; depends on multi head attention block and feedforward block (which is just relu followd by linear)
; TODO (requires multi-head attention block)

; softmax block
; (list n N) -> (list n N)
(define softmax-block
  (lambda ()
    (block softmax (list))))
; malt-included softmax here, NOT softmax-f

