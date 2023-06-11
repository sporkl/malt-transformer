#lang racket

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

(generate-arithmetic-sequence 0 1 5)
(generate-arithmetic-sequence 4 5 0)
(generate-arithmetic-sequence 4 5 4)

(define sequence-definitions
  (cartesian-product (range 500) (range 100) '(20)))

(define sequence-fragments
  (map
    (lambda (sd)
      (generate-arithmetic-sequence (car sd) (cadr sd) (caddr sd)))
    sequence-definitions))

(take sequence-fragments 20)

(length sequence-fragments)

(define sequence-string-list
  (map
    (lambda (s)
      (string-join (map number->string s) " "))
    sequence-fragments))

(take sequence-string-list 3)

(define sequences-string
  (string-join sequence-string-list "
"))

(display sequences-string (open-output-file "sequences.txt"))
