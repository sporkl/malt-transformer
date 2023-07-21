#lang racket

(require malt)

; n = # of words inputted to the model
; N = vocabulary size/length of one-hot-vectors which encode words
; d_model = dimension of embedding space for the model in general (outputted from each attention block)
; d_k = dimension of embedding space for queries and keys
; d_v = dimension of embedding space for values
; h = # heads per multi-head-attention block
; x, y, z = misc. variables

; DROPOUT

(require math/distributions)

(define dropout-0
  (λ (p)
    (let ((s (distribution-sample (bernoulli-dist p))))
      (λ (t)
        (s)))))

(define dropout
  (λ (p)
    (let ((scale (/ 1.0 p))
          (s* (ext1 (dropout-0 p) 0)))
      (λ (t)
        (λ (θ)
          (* scale (* (s* t) t)))))))

(define dropout-block
  (λ (p)
    (block
      (dropout p)
      (list))))

; dropout should go right after mha and feedforward (before layernorm)
; and right before values calculation in attention

; note: b/c dropout doesn't use any thetas, can provide another network without dropout blocks and is theta-compatible

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
      (let ([u ((mean-matrix m n) t)])
        (sqrt ((mean-matrix m n) (expt (- t u) 2)))))))

; normalization
; (list n d_model) -> (list n d_model)
(define normalize
  (lambda (m n)
    (lambda (t)
      (let ([u ((mean-matrix m n) t)]
            [σ ((stddev-matrix m n) t)])
        (/ (- t u) σ)))))

; normalization as a layer function
; this includes shift and scale after, as in the paper
(define normalize-layer
  (lambda (m n)
    (lambda (t)
      (lambda (θ)
        (let ([n ((normalize m n) t)])
          ((linear n) θ))))))

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
             [vals (*-2-1 V processed-scores)]
             ; DROPOUT GOES HERE
             )
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

; PARALLEL CONCAT BLOCK
; runs the same block h times in parallel. concatenates result along vectors

; concats a list of tensors along the vectors
(define concat-along-vectors
  (lambda (ts)
    (concat-along-vectors-helper (cdr ts) (car ts))))

(define concat-along-vectors-helper
  (lambda (ts a)
    (cond
      [(null? ts) a]
      [else
        (concat-along-vectors-helper
          (cdr ts)
          (concat (car ts) a))])))

; repeats x n times in a list
(define repeat
  (lambda (x n)
    (build-list
      n
      (lambda (_) x))))

; repeats a list l n times

(define repeat-list
  (lambda (l n)
    (apply append (repeat l n))))

; parallel layer

(define parallel-concat-layer
  (lambda (b h)
    (lambda (t)
      (lambda (theta)
        (concat-along-vectors
          (map-block
            b
            (repeat t h)
            theta
            '()))))))

; maps a block b over list of tensors ts using corresponding parameters in theta. (len theta) = (* (len (block-ls b)) (len ts))
(define map-block
  (lambda (b ts theta a)
    (cond
      [(null? ts) a]
      [else
        (map-block
          b
          (cdr ts)
          (refr theta (len (block-ls b)))
          (cons
            (((block-fn b) (car ts)) theta)
            a))])))

; parallel block

; TODO: switch back to tensor-based parallel approach

(define parallel-concat-block
  (lambda (b h)
    (block
      (parallel-concat-layer b h)
      (repeat-list (block-ls b) h))))

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
        (parallel-concat-block
          (attention-block d_model d_k d_v)
          h)
        (linear-block d_model (*-ρ h d_v))))))

; maked multi-head attention block
; (list n d_model) -> (list n d_model)
(define masked-multi-head-attention-block
  (lambda (n d_model d_k d_v h)
    (stack-blocks
      (list
        (parallel-concat-block
          (masked-attention-block n d_model d_k d_v)
          h)
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
        ; DROPOUT GOES HERE
        (normalize-block n d_model)
        (skip-block
          (feedforward-block d_model))
        ; DROPOUT GOES HERE
        (normalize-block n d_model)))))

; softmax block
; (list n N) -> (list n N)
(define softmax-block
  (lambda ()
    (block softmax (list))))
; malt-included softmax here, NOT softmax-f

(define log-block
  (lambda (m)
    (block
      (lambda (t)
        (lambda (theta)
          (writeln (shape t))
          (writeln m)
          t))
      (list))))

; TRANSFORMER

; here it is!
; n = 15
; N = 11
; d_model = 8
; d_k = 2
; d_v = 2
; h = 4
; batch size 1 (increase when get things working)
; dropout (p) = 0.2
; repeat 3 times
(define counter-transformer-network
  (stack-blocks
    (list
      (linear-block 11 8)
      (positional-encoding-block 15 8)
      (transformer-block 15 8 2 2 4)
      #| (transformer-block 15 8 2 2 4) |#
      #| (transformer-block 15 8 2 2 4) |#
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
        sequences-test-xs sequences-test-ys))))

(define test-transformer
  (lambda (network)
    (accuracy network sequences-test-xs sequences-test-ys)))

; train!

(define product (lambda (l) (foldl (lambda (x y) (* x y)) 1 l)))
(define sum-l (lambda (l) (foldl (lambda (x y) (+ x y)) 0 l)))
(sum-l (map product (block-ls counter-transformer-network))) ; with current params should be half the size of morse-fcn

(start-logging)
(train-and-test-transformer counter-transformer-network)

#| (test-transformer |#
#|   (model |#
#|     (block-fn counter-transformer-network) |#
#|     (init-theta (block-ls counter-transformer-network)))) |#
