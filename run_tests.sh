#!/bin/bash
# write the random generator for test cases in gen.cpp and compile it to gen
# compile working / brute force solution to working
# code to be tested compiled to a
UPPER_BOUND=10000

for((i = 1; i <= $UPPER_BOUND; ++i)); do
    echo $i; # only to see progress in testcase numbers
    ./gen $i > int
    ./a < int > out1
    ./working < int > out2
    diff -w out1 out2 || break
done