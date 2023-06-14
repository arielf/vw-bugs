VW = vw

FIXED_VW_ARGS = --noconstant -k --learning_rate 2 --progress 1
DATA = leaktest.vw
NN = 2
VW_NNARGS = --nn $(NN)
# Adding --multitask makes no difference (same bug)

all: default isolated-nn nn-leak

default:
	#
	# Verifying normal no --nn (thus no leak) operation
	#
	vw $(FIXED_VW_ARGS) -d $(DATA) 2>&1 | ./check-progress

isolated-nn:
	#
	# Verifying isolated name-spaces with --nn
	#
	grep 'always2' $(DATA) | \
	    vw $(FIXED_VW_ARGS) $(VW_NNARGS) -d /dev/stdin 2>&1 |./check-progress
	grep 'always5' $(DATA) | \
	    vw $(FIXED_VW_ARGS) $(VW_NNARGS) -d /dev/stdin 2>&1 |./check-progress

nn-leak:
	#
	# Triggering bug (data leak between name-spaces with --nn)
	#
	vw $(FIXED_VW_ARGS) $(VW_NNARGS) -d $(DATA) 2>&1 | ./check-progress
