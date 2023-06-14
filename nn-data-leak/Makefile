VW = vw

VWPARAMS = --noconstant -k --learning_rate 2 --progress 1
DATA = leaktest.vw
NN = 2

all: default isolated-nn nn-leak

default:
	#
	# Verifying normal no --nn (thus no leak) operation
	#
	vw $(VWPARAMS) -d $(DATA) 2>&1 | ./check-progress

isolated-nn:
	#
	# Verifying isolated name-spaces with --nn
	#
	grep 'always2' $(DATA) | \
	    vw $(VWPARAMS) --nn $(NN) -d /dev/stdin 2>&1 |./check-progress
	grep 'always5' $(DATA) | \
	    vw $(VWPARAMS) --nn $(NN) -d /dev/stdin 2>&1 |./check-progress

nn-leak:
	#
	# Triggering bug (data leak between name-spaces with --nn)
	#
	vw $(VWPARAMS) -d $(DATA) --nn $(NN) 2>&1 | ./check-progress
