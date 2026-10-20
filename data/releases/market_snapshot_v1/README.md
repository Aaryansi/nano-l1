This is the fixed numerical input for the corrected experiments. It contains
6,425 public Kalshi BTC 15-minute contract episodes, with 14 decision rows per
episode and five joined Binance spot features. Market opening times span
June 22 through August 29, 2026 UTC. The source API's moving history window
cannot guarantee the same membership on a later download.

`tickers.txt` identifies each episode in array order. `manifest.json` records
SHA-256 hashes, coverage, dimensions, and exact chronological split membership.
The normalizer is fitted on the training partition only. The snapshot preserves
the previously collected input; it does not repair missing quotes or establish
that simulated fills were available in a live market.

The default `reproduce.sh` uses this snapshot and verifies it before training.
Set `CORPUS` to evaluate a different corpus in a separately identified run.
The ingestion scripts remain available for collecting new data; those datasets
are new experiments and do not reproduce this snapshot's membership.
