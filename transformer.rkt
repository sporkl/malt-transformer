#lang racket

; transformer.rkt
; by Dmitri Volkov
; Implements a decoder-only transformer

(require malt)

; STANDARD BLOCKS

(define dense-block
  (lambda (n m)
    (block relu
      (list
        (list m n)
        (list m)))))

(define linear-block
  (lambda (n m)
    (block linear
      (list
        (list m n)
        (list m)))))

; softmax block
; (list n N) -> (list n N)
(define softmax-block
  (lambda ()
    (block softmax (list))))

; DROPOUT

(require math/distributions)

(declare-hyper p) ; p is dropout probability

(define dropout-0
  (λ (p)
    (let ((s (distribution-sample (bernoulli-dist p))))
      (λ (t)
        (s)))))

(define dropout
  (λ (t)
    (let ((scale (/ 1.0 p))
          (s* (ext1 (dropout-0 p) 0)))
      (* scale (* (s* t) t)))))

(define dropout-layer
  (lambda (t)
    (lambda (theta)
      (dropout t))))

(define dropout-block
  (λ ()
    (block
      dropout-layer
      (list))))

; SKIP

; skip block
(define skip-block
  (lambda (b)
    (block
      (lambda (t)
        (lambda (theta)
          (+ t (((block-fn b) t) theta))))
      (block-ls b))))

; PARALLEL CONCAT

; consider switching back to tensor-based parallel approach

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
(define repeat-append
  (lambda (l n)
    (apply append (repeat l n))))

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

; parallel block
(define parallel-concat-block
  (lambda (b h)
    (block
      (parallel-concat-layer b h)
      (repeat-append (block-ls b) h))))

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

(define normalize-block
  (lambda (n d_model)
    (block
      (normalize-layer n d_model)
      (list
        (list d_model d_model)
        (list d_model)))))

; ATTENTION

(define softmax-f
  (lambda (t)
    (let ((z (- t (max t))))
      (let ((expz (exp z)))
        (/ expz (sum expz))))))

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
             [vals (*-2-1 V processed-scores)]) ; consider add dropout after this line
        (sum-cols vals)))))

(define masked-attention-layer
  (lambda (n d_k)
    (lambda (t)
      (lambda (θ)
        (let
          ([Q ((linear t) θ)]
           [K ((linear t) (refr θ 2))]
           [V ((linear t) (refr θ 4))])
          ((masked-attention n d_k) Q K V))))))

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

; MULTI-HEAD ATTENTION

(define masked-multi-head-attention-block
  (lambda (n d_model d_k d_v h)
    (stack-blocks
      (list
        (parallel-concat-block
          (masked-attention-block n d_model d_k d_v)
          h)
        (linear-block d_model (*-ρ h d_v))))))

; TRANSFORMER

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

(define transformer-block-with-dropout
  (lambda (n d_model d_k d_v h)
    (stack-blocks
      (list
        (skip-block
          (masked-multi-head-attention-block n d_model d_k d_v h))
        (dropout-block)
        (normalize-block n d_model)
        (skip-block
          (feedforward-block d_model))
        (dropout-block)
        (normalize-block n d_model)))))

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

; POSITIONAL ENCODING

; positional encoding block for learned positional encoding
; (list n d_model) -> (list n d_model)
(define positional-encoding-block
  (lambda (n d_model)
    (block
      (lambda (t)
        (lambda (θ)
          (+ t (ref θ 0))))
      (list (list n d_model)))))

; TRANSFORMER ARCHITECTURE

; repeats a block b in a row n times
(define repeat-block
  (lambda (b n)
    (stack-blocks (repeat b n))))

; n: length of input
; N: number of potential values for input
; d_model: dimensionality of the model (how long are the vectors used throughout)
; d_k: dimensionality of the keys and queries
; d_v: dimensionality of the values
; h: number of heads for each attention
; r: number of times to repeat transformer block

(define transformer-network-with-dropout
  (lambda (n N d_model d_k d_v h r)
    (stack-blocks
      (list
        (linear-block N d_model)
        (positional-encoding-block n d_model) 
        (repeat-block
          (transformer-block-with-dropout n d_model d_k d_v h)
          r)
        (linear-block d_model N)
        (softmax-block)))))

(define transformer-network
  (lambda (n N d_model d_k d_v h r)
    (stack-blocks
      (list
        (linear-block N d_model)
        (positional-encoding-block n d_model) 
        (repeat-block
          (transformer-block n d_model d_k d_v h)
          r)
        (linear-block d_model N)
        (softmax-block)))))

; TRAINING AND TESTING

; train a network
(define train-network
  (lambda (network-for-training network theta-shapes xs ys) ; different train and test architectures to remove dropout
    (model network
      (adam-gradient-descent
        (sampling-obj
          ((with-recording l2-loss)
            network-for-training)
           xs ys)
        (init-theta theta-shapes)))))

; just use accuracy to test

; n = 15
; N = 11
; d_model = 8
; d_k = 2
; d_v = 2
; h = 4
; r = 3

(define counter-for-training
  (transformer-network-with-dropout 15 11 8 2 2 4 3))

(define counter
  (transformer-network 15 11 8 2 2 4 3))

; get the data
(require "data/arithmetic-sequences/arithmetic-sequences.rkt")

(define train-counter
  (λ ()
    (with-hypers ; TODO: use grid search
      ((alpha 0.0005)
       (revs 1) ; change to 20000
       (batch-size 1)
       (mu 0.9)
       (beta 0.999)
       (p 0.2))
      (train-network
        (block-fn counter-for-training)
        (block-fn counter)
        (block-ls counter)
        sequences-train-xs sequences-train-ys))))

(define test-counter
  (lambda (network)
    (accuracy network sequences-test-xs sequences-test-ys)))

(define train-and-test-counter
  (lambda ()
    (let ((trained-counter (train-counter)))
      (begin
        ; TODO: save theta to disk here
        (test-counter trained-counter)))))

#| (start-logging) |#
#| (train-and-test-counter) |#
