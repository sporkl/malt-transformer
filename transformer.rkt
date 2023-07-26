#lang racket

; transformer.rkt
; by Dmitri Volkov
; Implements a decoder-only transformer

(require malt)

; LOG BLOCK

; useful for debugging

(define log-block
  (lambda (f) ; f should take t as argument
    (block
      (lambda (t)
        (lambda (theta)
          (writeln (f t))
          t))
      (list))))

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

(define dropout-0-ρ
  (λ (s)
    (λ (t)
      (s))))

(define dropout-0-∇
  (λ (r z) z))

(define dropout-0
  (λ (prob)
    (let ((s (distribution-sample (bernoulli-dist prob))))
      (ext1 (prim1 (dropout-0-ρ s) dropout-0-∇) 0))))

(define dropout-ext1
  (λ (prob)
    (λ (t)
      (let ((noise-generator (dropout-0 prob))
            (scale (/ 1.0 prob)))
        (let ((noise (noise-generator t)))
          (* (* scale noise) t))))))

(define dropout
  (λ (prob)
    (let ((op (dropout-ext1 prob)))
      (λ (t)
        (λ (θ)
          (op t))))))

(define dropout-block
  (λ (prob)
    (block
      (dropout prob)
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

; parallel-concat block
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
      (lambda (theta)
        (let ([n ((normalize m n) t)])
          ((linear n) theta))))))

(define normalize-block
  (lambda (n d_model)
    (block
      (normalize-layer n d_model)
      (list
        (list d_model d_model)
        (list d_model)))))

; ATTENTION

(define ghostmax-f
  (lambda (t)
    (let ((z (- t (max t))))
      (let ((expz (exp z)))
        (/ expz (+ (sum expz) 1)))))) ; + 1 in denominator is very recent research! theoretically better, but lacking applied evidence

(define make-future-mask
  (lambda (n)
    (build-tensor
      (list n n)
      (lambda (p)
        (let ([x (ref p 1)]
              [y (ref p 0)])
          (cond
            [(> x y) -inf.0]
            [else 0]))))))

(define masked-attention
  (lambda (n d_k)
    (lambda (Q K V)
      (let* ([scores (dot-product-2-1 K Q)]
             [masked-scores
               (+ scores
                  (make-future-mask n))]
             [processed-scores
               (ghostmax-f
                 (/ masked-scores (sqrt d_k)))]
             [vals (*-2-1 V processed-scores)]) ; consider add dropout after this line
        (sum-cols vals)))))

(define masked-attention-layer
  (lambda (n d_k)
    (lambda (t)
      (lambda (theta)
        (let*
          ([Q ((linear t) theta)]
           [K ((linear t) (refr theta 2))]
           [V ((linear t) (refr theta 4))]
           )
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
        (linear-block (*-ρ h d_v) d_model)
        ))))

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
        (dropout-block p)
        (normalize-block n d_model)
        (skip-block
          (feedforward-block d_model))
        (dropout-block p)
        (normalize-block n d_model)
        ))))

(define transformer-block
  (lambda (n d_model d_k d_v h)
    (stack-blocks
      (list
        (skip-block
          (masked-multi-head-attention-block n d_model d_k d_v h))
        (normalize-block n d_model)
        (skip-block
          (feedforward-block d_model))
        (normalize-block n d_model)
        ))))

; POSITIONAL ENCODING

; positional encoding block for learned positional encoding
; (list n d_model) -> (list n d_model)
(define positional-encoding-block
  (lambda (n d_model)
    (block
      (lambda (t)
        (lambda (theta)
          (+ t (ref theta 0))))
      (list (list n d_model)))))

; TAIL BLOCK

(define make-tail-mask
  (lambda (h w)
    (build-tensor
      (list h w)
      (lambda (idx)
        (cond
          [(= (car idx) (sub1 h)) 1.0]
          [else 0.0])))))

(define d-tail-2
  (lambda (t)
    (sum-cols
      (* t (make-tail-mask (tlen (tref t 0)) (tlen (tref (tref t 0) 0))))))) ; t here includes batch dimension :(

; TODO: d-tail-2 implementation that doesn't require batch dimension. maybe if take 2 arguments at start

(define tail-layer
  (lambda (t)
    (lambda (theta)
      (d-tail-2 t))))

(define tail-block
  (lambda ()
    (block tail-layer (list))))

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
        (tail-block)
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
        (tail-block)
        (linear-block d_model N)
        (softmax-block)))))

; TRAINING AND TESTING

; saves a theta to a filename
(define print-theta
  (lambda (theta)
    (let ((tpl (max-tensor-print-length)))
      (max-tensor-print-length 0)
      (if #f (writeln theta) '()) ; change to #f to avoid printing theta
      (max-tensor-print-length tpl))
    theta))

; train a network
(define train-network
  (lambda (network-for-training network theta-shapes xs ys) ; different train and test architectures to remove dropout
    (model
      network
      (print-theta
        (adam-gradient-descent
          (sampling-obj
            ((with-recording l2-loss)
             network-for-training)
            xs ys)
          (init-theta theta-shapes))))))

; just use accuracy to test

; n = 8
; N = 11
; d_model = 3
; d_k = 2
; d_v = 2
; h = 2
; r = 1

; use this for training once flat-tensors dropout works
(define counter-for-training
  (lambda () ; needed to prevent from loading dropout probability hyper
    (transformer-network-with-dropout 8 11 3 2 2 2 1)))

(define counter
  (lambda ()
    (transformer-network 8 11 3 2 2 2 1)))

; get the data
(require "data/arithmetic-sequences/arithmetic-sequences.rkt")

(define find-counter-hypers
  (lambda ()
    (grid-search
      (lambda (a) (> a .5)) ; just need to find something that works a bit, then use more epochs
      ((revs 100 500 1000 2000 4000 10000)
       (batch-size 2 4 8)
       (alpha 0.01 0.005 0.0005 0.0001)
       (mu 0.9)
       (beta 0.999)
       (p 0.1)
       )
      (writeln (list revs alpha batch-size))
      (let ((acc
              (accuracy
                (train-network
                  (block-fn (counter-for-training))
                  (block-fn (counter))
                  (block-ls (counter))
                  sequences-train-xs sequences-train-ys)
                sequences-test-xs sequences-test-ys)))
        (write "Acc: ")
        (writeln acc)
        acc))))

; tried and cut short with 1 transformer block, but best I got was 0.33 with revs=500, batch size=2, alpha = 0.01
; I think only having 1 attention block is limiting what it can do.

(define train-counter
  (lambda ()
    (with-hypers ; TODO: use grid search
      ((alpha 0.001)
       (revs 1) ; change to 20000
       (batch-size 1) ; note: batch-size != context length otherwise + gets confused
       (mu 0.9)
       (beta 0.999)
       (p 0.9)) ; opposite of pytorch definition of dropout
      (train-network
        (block-fn (counter-for-training))
        (block-fn (counter))
        (block-ls (counter))
        sequences-train-xs sequences-train-ys))))

(define test-counter
  (lambda (network)
    (accuracy network sequences-test-xs sequences-test-ys)))

(define train-and-test-counter
  (lambda ()
    (let ((trained-counter (train-counter)))
      (begin
        (test-counter trained-counter)))))

(max-tensor-print-length 0)

; TODO: have train input a starting theta and output a theta

(start-logging)
#| (find-counter-hypers) |#
(train-and-test-counter)
