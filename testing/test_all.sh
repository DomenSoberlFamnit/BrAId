#!/bin/bash

PROGRAM="python"

for i in 1 2 3 4 5 6 7 8 9 10
do
	echo "Testing experiment $i"
	$PROGRAM classify_test.py $i
done
