# vw data-leak with the `--nn` option bug:

This repo demonstrates a bug in vw, found on 2023-06-13,
but verified to exist as far back as 2014 (~9 years ago)

## To reproduce, run `make`

    make

## Explanation

The dataset `leaktest.vw`:

  - Has two separate name-spaces (`always2` and `always5`)
  - The name-spaces have no features in common (due to name-space separation)
  - The label of each example is set to a constant (one per each name space)

The Makefile has 3 targets:

  - Vanilla (no `--nn` used) to show what's expected
  - Using `--nn` on each name space separately (two runs, one for each
    isolated name-space)
  - Use `--nn` on both name spaces in one unified training data-set

All the `vw` runs are with `--noconstant` and no caching `-k`.

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

Here's how success looks like:

```
#
# Verifying normal no --nn (thus no leak) operation
#
vw --noconstant -k --learning_rate 2 --progress 1 -d leaktest.vw 2>&1 |
     ./check-progress
All OK! (predicted values monotonically approaching label)
        P2: 0.0000 1.7293 1.9627 1.9949 1.9993 1.9999
        P5: 0.0000 2.7532 3.9168 4.4706 4.7404 4.8726
#
# Verifying isolated name-spaces with --nn
#
grep 'always2' leaktest.vw | \
    vw --noconstant -k --learning_rate 2 --progress 1 --nn 2 -d /dev/stdin 2>&1 |
    ./check-progress
All OK! (predicted values monotonically approaching label)
        P2: 0.0000 1.4939 2.0000 2.0000 2.0000 2.0000
        P5:

grep 'always5' leaktest.vw | \
    vw --noconstant -k --learning_rate 2 --progress 1 --nn 2 -d /dev/stdin 2>&1 |./check-progress
All OK! (predicted values monotonically approaching label)
        P2:
        P5: 0.0000 2.0418 4.3590 4.8295 4.9513 4.9855
```

## What fails?

We get multiple failures (violations of expected invariants)
for the 3rd Makefile target:

  - nn-leak

There seem to be some data leak (unwanted interaction) between the two name-spaces.

Each of the two streams breaks the other through some suspected shared
feature weight.

This causes the SGD update go against the correct gradient.
The `always2` stream pushes the `always5` down, while
the `always5` pushes the `always2` stream up.

Here's the failure case in detail:
```
#
# Triggering bug (data leak between name-spaces with --nn)
#
vw --noconstant -k --learning_rate 2 --progress 1 -d leaktest.vw --nn 2 2>&1 |
     ./check-progress
constant label=2: example-no=2  predicted=2.3587 exceeds max (2):
 1.6483 2.3587 1.4182 3.2389 0.3825 2.3662
constant label=2: example-no=3  predicted=1.4182 is non-monotonic!
 1.6483 2.3587 1.4182 3.2389 0.3825 2.3662
constant label=2: example-no=4  predicted=3.2389 exceeds max (2):
 1.6483 2.3587 1.4182 3.2389 0.3825 2.3662
constant label=2: example-no=5  predicted=0.3825 is non-monotonic!
 1.6483 2.3587 1.4182 3.2389 0.3825 2.3662
constant label=2: example-no=6  predicted=2.3662 exceeds max (2):
 1.6483 2.3587 1.4182 3.2389 0.3825 2.3662
constant label=5: example-no=5  predicted=4.3549 is non-monotonic!
 0.0000 2.1334 4.2154 4.7311 4.3549 4.3113
constant label=5: example-no=6  predicted=4.3113 is non-monotonic!
 0.0000 2.1334 4.2154 4.7311 4.3549 4.3113

```
