# vw data-leak with the `--nn` option:

This repo demonstrates a bug in vw, found on 2023-06-13,
but verified to exist as far back as 2014 (~9 years ago)

The bug is that using `--nn` causes an unexpected data-leak
and the update goes the wrong way (against the desired gradient
towards minimum loss).

## To reproduce, run `make`

    make

## Explanation

### The dataset `leaktest.vw`:

  - Has two separate name-spaces (`always6` and `always7`)
  - The name-spaces have no features in common (due to name-space separation)
  - The label of each example is set to a constant (one per each name space)

Here's a copy of the dataset:
```
6.0 time0/always6|always6  f1:1
7.0 time0/always7|always7  f1:1
6.0 time1/always6|always6  f1:1
7.0 time1/always7|always7  f1:1
6.0 time2/always6|always6  f1:1
7.0 time2/always7|always7  f1:1
6.0 time3/always6|always6  f1:1
7.0 time3/always7|always7  f1:1
6.0 time4/always6|always6  f1:1
7.0 time4/always7|always7  f1:1
```

### The Makefile has 3 targets:

  - Vanilla (no `--nn` used) to show what's expected
  - Using `--nn` on each name space separately (two runs, one for each
    isolated name-space)
  - Use `--nn` on both name spaces in one unified training data-set

All the `vw` runs are one pass, with `--noconstant` and no caching.

## What is expected?

Since the two name spaces:

  - Represent two independent streams with no features in common
  - With a constant label

and since the runs are with `--noconstant` and no caching,
I expect the predicted value to converge monotonically towards
the respective constant label in each of the two cases.

Each name space should have its own convergence towards its constant label.

Indeed, we get the expected result for the 1st two Makefile targets:

  - default
  - isolated-nn

### Here's how success looks like:

Simple case, no `--nn` used:
```
$ make default
#
# Verifying normal no --nn (thus no leak) operation
#
vw --noconstant -k --learning_rate 10 --progress 1 -d leaktest.vw 2>&1 | \
    ./check-progress
All OK! (predicted values monotonically approaching label)
        P6: 0.0000 4.8666 5.7797 5.9571 5.9917
        P7: 0.0000 5.3224 6.5818 6.8955 6.9739
```

Isolated name-spaces (2 runs with `--nn`):
```
$ make isolated-nn
#
# Verifying isolated name-spaces with --nn
#
grep -- "always6" leaktest.vw | \
    vw --noconstant -k --learning_rate 10 --progress 1 --nn 1 -d /dev/stdin 2>&1 | \
        ./check-progress
All OK! (predicted values monotonically approaching label)
        P6: 0.0000 5.3137 6.0000 6.0000 6.0000
grep -- "always7" leaktest.vw | \
    vw --noconstant -k --learning_rate 10 --progress 1 --nn 1 -d /dev/stdin 2>&1 | \
        ./check-progress
All OK! (predicted values monotonically approaching label)
        P7: 0.0000 5.7712 7.0000 7.0000 7.0000
```

## What fails?

We get multiple failures (violations of expected invariants)
for the 3rd Makefile target:

  - nn-leak

There seem to be some data leak (unwanted interaction) between the two name-spaces.

Each of the two streams breaks the other through some suspected shared
feature weight.

This causes the SGD update go against the correct gradient.
The `always6` stream pushes the `always7` down, while
the `always7` pushes the `always6` stream up.

Here's the failure case in detail:
```
$ make nn-leak
#
# Triggering bug (data leak between name-spaces with --nn)
#
vw --noconstant -k --learning_rate 10 --progress 1 --nn 1 -d leaktest.vw 2>&1 | \
    ./check-progress
constant label=7: example-no=4  predicted=6.1166 is non-monotonic!
 4.8666 6.1036 6.1701 6.1166 6.0900
constant label=7: example-no=5  predicted=6.0900 is non-monotonic!
 4.8666 6.1036 6.1701 6.1166 6.0900
constant label=6: example-no=2  predicted=7.0000 exceeds max (6):
 0.0000 7.0000 7.0000 7.0000 7.0000
constant label=6: example-no=3  predicted=7.0000 exceeds max (6):
 0.0000 7.0000 7.0000 7.0000 7.0000
constant label=6: example-no=4  predicted=7.0000 exceeds max (6):
 0.0000 7.0000 7.0000 7.0000 7.0000
constant label=6: example-no=5  predicted=7.0000 exceeds max (6):
 0.0000 7.0000 7.0000 7.0000 7.0000
```

## github issue

For reference, this is the issue I opened in the official vw repo:

[VowpalWabbit/vowpal_wabbit#4614](https://github.com/VowpalWabbit/vowpal_wabbit#4614)


## Possible cause + desired outcome

My guess is that the leak happens through the full-connectivity of the
features via the hidden layer.

The full connectivity is a done-deal (imposed at the start of run
by the fact we want a fully-connected NN.)

So it seems to me that the SGD update should somehow skip any updates to
weights that have nothing to do with the ones in the example.

IOW: the skips should be in run-time (rather than initialization time)
and should update only those target feature-nodes that are present
in the current example (and/or namespace).

Ideally, this skip vs non-skip (current default) should be controlled by
a `vw` CLI switch. Possible proposed names:

     --respect_namespaces
     --restricted_update
