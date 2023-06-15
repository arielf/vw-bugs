VW = vw

FIXED_VW_ARGS = --noconstant -k --learning_rate 10 --progress 1
DATA = leaktest.vw
NN = 1
VW_NNARGS = --nn $(NN)
# Adding --multitask makes no difference (same bug)

NS1=$(shell grep -Eo -- 'always[0-9]' $(DATA) | head -1)
NS2=$(shell grep -Eo -- 'always[0-9]' $(DATA) | tail -1)

all: check default isolated-nn nn-leak

check:
	# -- Name-spaces are:  1:$(NS1)  2:$(NS2)

default:
	#
	# Verifying normal no --nn (thus no leak) operation
	#
	vw $(FIXED_VW_ARGS) -d $(DATA) 2>&1 | \
	    ./check-progress

isolated-nn:
	#
	# Verifying isolated name-spaces with --nn
	#
	grep -- "$(NS1)" $(DATA) | \
	    vw $(FIXED_VW_ARGS) $(VW_NNARGS) -d /dev/stdin 2>&1 | \
	    	./check-progress
	grep -- "$(NS2)" $(DATA) | \
	    vw $(FIXED_VW_ARGS) $(VW_NNARGS) -d /dev/stdin 2>&1 | \
	    	./check-progress

nn-leak:
	#
	# Triggering bug (data leak between name-spaces with --nn)
	#
	vw $(FIXED_VW_ARGS) $(VW_NNARGS) -d $(DATA) 2>&1 | \
	    ./check-progress
