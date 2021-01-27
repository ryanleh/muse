<h1 align="center">Muse</h1>

___Muse___ is a Python, C++, and Rust library for **Secure Convolutional Neural Network Inference for Malicious Clients**. 

**WARNING:** This is an academic proof-of-concept prototype, and in particular has not received careful code review. This implementation is NOT ready for production use.

## Overview

This library implements the components of a cryptographic system for efficient client-malicious inference on general convolutional neural networks.

**The end-to-end protocol is not yet implemented**. In particular, the following components are missing:
* TopGear zero-knowledge proof integration (latency costs are currently simulated)
* The various components do not currently run in parallel

The rest of this README will walk through being able to run the various components of Muse which will allows someone to reproduce the results in Tables 3/4 and Figures 8 9, and 10


## Directory structure

This repository contains several folders that implement the different building blocks of Delphi. The high-level structure of the repository is as follows.
* [`python`](python): Example Python scripts for performing neural architecture search (NAS)

* [`rust/algebra`](rust/algebra): Rust crate that provides finite fields

* [`rust/crypto-primitives`](rust/crypto-primitives): Rust crate that implements some useful cryptographic primitives

* [`rust/experiments`](rust/experiments): Rust crate for running latency, bandwidth, throughput, accuracy, and memory usage experiments

* [`rust/neural-network`](rust/neural-network): Rust crate that implements generic neural networks

* [`rust/protocols`](rust/protocols): Rust crate that implements cryptographic protocols

* [`rust/protocols-sys`](rust/crypto-primitives): Rust crate that provides the C++ backend for Delphi's pre-processing phase and an FFI for the backend

In addition, there is a  [`rust/bench-utils`](rust/bench-utils) crate which contains infrastructure for benchmarking. This crate includes macros for timing code segments and is used for profiling the building blocks of Delphi.

## Setup

### Local setup

If you'd like to compile and run experiments locally, please first install [rustup](https://rustup.rs/), and then install the latest Rust nightly as follows:
```bash
rustup install nightly
```
Note that this is only necessary if you are *not* using the AWS image.

### Instance setup

### Experiments

First, `cd experiments` to switch to the `experiments` folder.

#### Table 3

##### Correlation generator

To measure the cost of correlation generation, first build the relevant binaries:
```bash
cargo build --release acg-client --all-features;
cargo build --release acg-server --all-features;
```

Then, in separate windows, execute these binaries with the relevant options:
```bash
# Start server:
env RAYON_NUM_THREADS=1 env BENCH_OUTPUT_FILE="./acg_times.csv" cargo run --release --all-features acg-server <0/1> <port> > /dev/null 2>&1 &;
# Start client:
env RAYON_NUM_THREADS=1 cargo run --release --all-features acg-client <0/1> <server_ip> <server_port> > /dev/null 2>&1;
```
This will write out the time taken by the correlation generator to `./acg_times.csv`

##### Garbling

To measure the cost of garbling the ReLU circuits, first build the relevant binaries:
```bash
cargo build --release garbling-client --all-features;
cargo build --release garbling-server --all-features;
```

Then, in separate windows, execute these binaries with the relevant options:
```bash
# Start server:
env RAYON_NUM_THREADS=1 env BENCH_OUTPUT_FILE="./garbling_times.csv" cargo run --release --all-features garbling-server <0/1> <port> > /dev/null 2>&1 &;
# Start client:
env RAYON_NUM_THREADS=1 cargo run --release --all-features garbling-client <0/1> <server_ip> <server_port> > /dev/null 2>&1;
```
This will write out the time taken by the correlation generator to `./garbling_times.csv`
To obtain the final time, run `cat garbling_times.csv | grep "Garbling"`

##### CDS

To measure the cost of the CDS protocol, first build the relevant binaries:
```bash
cargo build --release cds-client --all-features;
cargo build --release cds-server --all-features;
```

Then, in separate windows, execute these binaries with the relevant options:
```bash
# Start server:
env RAYON_NUM_THREADS=1 env BENCH_OUTPUT_FILE="./cds_times.csv" cargo run --release --all-features cds-server <0/1> <port> > /dev/null 2>&1 &;
# Start client:
env RAYON_NUM_THREADS=1 cargo run --release --all-features cds-client <0/1> <server_ip> <server_port> > /dev/null 2>&1;
```
This will write out the time taken by the correlation generator to `./cds_times.csv`
To obtain the final time, run `cat cds_times.csv | grep "CDS Protocol"`

##### Online phase

XXX

## License

Muse is licensed under either of the following licenses, at your discretion.

 * Apache License Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
