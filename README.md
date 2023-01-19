## measure_peering.py

Tool for conducting advertisements via the PEERING testbed. Integration with VULTR.

Two main sub-projects are PAINTER and sparse advertisements work. Sparse advertisements is under construction, PAINTER is more-or-less fully developed.


### PAINTER
Usage: python measure_peering.py --system vultr --mode [measure_prepainter/conduct_painter]

#### measure_prepainter

Goal is to obtain measurements necessary for calculating a PAINTER adverisement solution. Roughly these are
1. default anycast latencies
2. latencies from all clients to all ingresses

#### conduct_painter

Given an advertisement strategy (a set of (prefix, ingress) tuples), executes it and reports latency all clients achieve. Goal is to assess how well a PAINTER strategy actually does in the wild.
