#lang racket

(require malt/base-no-overrides)
(require racket/random)

(define n 15)
(define N 11)

; generates an arithmetic sequence starting at start, stepping by step, and of size size.
(define generate-arithmetic-sequence
  (lambda (start step size)
    (cond
      [(eqv? size 0) '()]
      [else
        (cons start
              (generate-arithmetic-sequence
                (+ start step)
                step
                (sub1 size)))])))

(define sequence-definitions
 (cartesian-product '(0 1 2 3 5 7 11) (range 100) '(10)))

(define sequence-fragments
  (map
    (lambda (sd)
      (generate-arithmetic-sequence (car sd) (cadr sd) (caddr sd)))
    sequence-definitions))

#| (take sequence-fragments 20) |#

#| (length sequence-fragments) |#

(define sequence-string-list
  (map
    (lambda (s)
      (string-join (map number->string s) " "))
    sequence-fragments))

#| (define sequences-string |#
#|   (string-join sequence-string-list " |#
#| ")) |#

(define substrings-of-len
  (lambda (s l)
    (sol-helper s l 0)))

(define sol-helper
  (lambda (s l i)
    (cond
      [(> i (- (string-length s) l)) '()]
      [else
        (cons
          (substring s i (+ i l))
          (sol-helper s l (add1 i)))])))

; first n characters are input data, last 1 is output
(define processed-sequences
  (set->list
    (list->set
      (flatten
        (map
          (lambda (x)
            (substrings-of-len x (add1 n)))
          sequence-string-list)))))

; has N clauses
(define char-lookup
  (lambda (c)
    (match c
      [#\0 0]
      [#\1 1]
      [#\2 2]
      [#\3 3]
      [#\4 4]
      [#\5 5]
      [#\6 6]
      [#\7 7]
      [#\8 8]
      [#\9 9]
      [#\space 10])))

(define string->numeric
  (lambda (s)
    (map char-lookup
         (string->list s))))

(define numberized-sequences (map string->numeric processed-sequences))

(define one-hot-tensor
  (lambda (i l)
    (build-tensor (list l)
      (lambda (p)
        (cond
          [(eqv? (car p) i) 1]
          [else 0])))))

(define one-hotted-sequences
  (map
    (lambda (l)
      (map
        (lambda (x)
          (one-hot-tensor x N))
        l))
    numberized-sequences))

(define pair-x-y-sequences
  (map
    (lambda (ts)
      (cons
        (list->tensor (take ts n))
        (ref ts n)))
    one-hotted-sequences))

(random-seed 231)

(define test-sequences
  (random-sample
    pair-x-y-sequences
    (floor (/ (length pair-x-y-sequences) 7)))) ; roughly 15 percent

(define train-sequences
  (remove* test-sequences pair-x-y-sequences))

(define sequences-test-xs
  (list->tensor
    (map
      (lambda (x) (car x))
      test-sequences)))

(define sequences-test-ys
  (list->tensor
    (map
      (lambda (x) (cdr x))
      test-sequences)))

(define sequences-train-xs
  (list->tensor
    (map
      (lambda (x) (car x))
      train-sequences)))

(define sequences-train-ys
  (list->tensor
    (map
      (lambda (x) (cdr x))
      train-sequences)))

(provide sequences-test-xs sequences-test-ys
         sequences-train-xs sequences-train-ys)
