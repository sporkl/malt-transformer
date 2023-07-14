#lang racket

(require malt)

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

(define parallel-concat-block
  (lambda (b h)
    (block
      (parallel-concat-layer b h)
      (repeat-list (block-ls b) h))))

(define xs (tensor (tensor 1 2) (tensor 3 4) (tensor 5 6)))
(define ys (tensor 1 2 3))

; relu/dense block
(define dense-block
  (lambda (n m)
    (block relu
      (list
        (list m n)
        (list m)))))

; define network

(define network
  (stack-blocks
    (list
      (parallel-concat-block
        (dense-block 2 2)
      2)
      (dense-block 1 4))))

; train and test funcs

(define train
  (λ (network)
    (with-hypers ; TODO: use grid search
      ((alpha 0.0005)
       (revs 10) ; change to 20000
       (batch-size 2)
       (mu 0.9)
       (beta 0.999))
      (trained (block-fn network) (block-ls network)))))

(define trained
  (λ (transformer theta-shapes)
    (model transformer
      (adam-gradient-descent
        (sampling-obj
          ((with-recording l2-loss)
            transformer)
           xs ys)
        (init-theta theta-shapes)))))

(define train-and-test
  (λ (network)
    (fprintf (current-error-port) "Accuracy: ~a~%"
      (accuracy
        (train network)
        xs ys))))

; train!

(start-logging)
(train-and-test network)




