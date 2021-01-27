<h1 align="center">Muse</h1>

___Muse___ is a Python, C++, and Rust library for **Secure Convolutional Neural Network Inference for Malicious Clients**. 

**WARNING:** This is an academic proof-of-concept prototype, and in particular has not received careful code review. This implementation is NOT ready for production use.

## Overview

This library implements the components of a cryptographic system for efficient client-malicious inference on general convolutional neural networks.

**The end-to-end protocol is not yet implemented**. In particular, the following components are missing:
* TopGear zero-knowledge proof integration (latency costs are currently simulated)
* Components are not currently run in parallel

The rest of this README will walk through running experiments for the various components of Muse to reproduce the results in Table 3 and Figures 8, 9, and 10.

Note that the implementation has changed slightly since the submission, so some of the numbers may be off by a small margin. Additionally, these micro-benchmarks do not include the communication cost of the zero-knowledge proofs.

## Directory structure

This repository contains several folders that implement the different building blocks of MUSE. The high-level structure of the repository is as follows.
* [`python`](python): Example Python scripts for performing neural architecture search (NAS)

* [`rust/algebra`](rust/algebra): Rust crate that provides finite fields

* [`rust/crypto-primitives`](rust/crypto-primitives): Rust crate that implements some useful cryptographic primitives

* [`rust/experiments`](rust/experiments): Rust crate for running latency, bandwidth, throughput, accuracy, and memory usage experiments

* [`rust/neural-network`](rust/neural-network): Rust crate that implements generic neural networks

* [`rust/protocols`](rust/protocols): Rust crate that implements cryptographic protocols

* [`rust/protocols-sys`](rust/crypto-primitives): Rust crate that provides the C++ backend for MUSE's pre-processing phase and an FFI for the backend

In addition, there is a  [`rust/bench-utils`](rust/bench-utils) crate which contains infrastructure for benchmarking. This crate includes macros for timing code segments and is used for profiling the building blocks of MUSE.

## Setup

### Local setup

If you'd like to compile and run experiments locally, please first install [rustup](https://rustup.rs/), and then install the latest Rust nightly as follows:
```bash
rustup install nightly
```
Note that this is only necessary if you are *not* using the AWS image.

### Instance setup

To enable running our code on AWS EC2 machines, we have created an Amazon Machine Image (AMI) that already contains an installation of Rust, as well as a copy of this code.

The ID of this AMI is ami-01421496a8ce0c211.

To set up an EC2 machine with our AMI, follow the instructions outlined [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/launching-instance.html), and, in Step 1, enter the AMI ID in the search bar and select "Community AMI".

To replicate our results from the paper, both the client and the server must be c5.9xlarge instances, and the region for the client machine should be us-west-1, while the region for the server should be us-west-2.

### Experiments

First, make sure the repository is up to date (we will push any bug fixes) via `git pull origin main`.  Next, run `cd rust/experiments` to switch to the crate containing our benchmarks.

#### Table 3/Figure 10

##### Authenticated correlations generator (ACG)

To measure the cost of the ACG, first build the relevant binaries:
```bash
cargo build +nightly --release acg-client --all-features;
cargo build +nightly --release acg-server --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --release --all-features --bin acg-server -- -m <0/1> 2>/dev/null > "./acg_time.txt"
# On the client instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --release --all-features --bin acg-client -- -m <0/1> -i <server_ip> 2>/dev/null > "./acg_time.txt"
```
This will write out a trace of execution times and bandwidth used to `./acg_time.txt`.

Note that the `-m` flag controls which model architecture is used: MNIST (0) or MiniONN (1).

##### Garbling

To measure the cost of garbling the ReLU circuits, first build the relevant binaries:
```bash
cargo build +nightly --release garbling-client --all-features;
cargo build +nightly --release garbling-server --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --release --all-features --bin garbling-server -- -m <0/1> 2>/dev/null > "./garbling_time.txt"
# On the client instance: 
env RAYON_NUM_THREADS=2 cargo +nightly run --release --all-features --bin garbling-client -- -m <0/1> -i <server_ip> 2>/dev/null > "./garbling_time.txt"
```
This will write out a trace of execution times and bandwidth used to `./garbling_time.txt`.

##### Triple Generation

To measure the cost of triple generation for the CDS protocol, first build the relevant binaries:
```bash
cargo build +nightly --release triples-gen-client --all-features;
cargo build +nightly --release triples-gen-server --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=6 cargo run --release --all-features --bin triples-gen-server -- -m <0/1> 2>/dev/null > "./triples_times.txt";
# On the client instance:
env RAYON_NUM_THREADS=6 cargo run --release --all-features --bin triples-gen-client -- -m <0/1> -i <server_ip> 2>/dev/null > "./triples_time.txt"
```
This will write out a trace to `./triples_time.txt`. Note that the results from Figure 10 can be reproduced by varying the number of threads in the `RAYON_NUM_THREADS` environment variable, and additionally including the `-n 10000000` flag.

##### Input Authentication

To measure the cost of input sharing for the CDS protocol, first build the relevant binaries:
```bash
cargo build +nightly --release input-auth-client --all-features;
cargo build +nightly --release input-auth-server --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=3 cargo run --release --all-features --bin input-auth-server -- -m <0/1> 2>/dev/null > "./input_auth_times.csv"
# On the client instance:
env RAYON_NUM_THREADS=3 cargo run --release --all-features --bin input-auth-client -- -m <0/1> -i <server_ip> 2>/dev/null > "./input_auth_time.txt"
```
This will write out a trace to `./input_auth_time.txt`.

##### CDS Evaluation

To measure the cost of evaluating the CDS protocol, first build the relevant binaries:
```bash
cargo build +nightly --release cds-client --all-features;
cargo build +nightly --release cds-server --all-features;
```

Then, execute these commands to run the experiment:
```bash
# On the server instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --release --all-features --bin cds-server -- -m <0/1> 2>/dev/null > "./cds_time.csv"
# On the client instance:
env RAYON_NUM_THREADS=2 cargo +nightly run --release --all-features --bin cds-client -- -m <0/1> -i <server_ip> 2>/dev/null > "./cds_time.txt"
```
This will write out a trace of execution times to  `./cds_time.txt`.

##### Online phase

To measure the cost of the online phase, first build the relevant binaries.
(The code examples show how to do this for MNIST; for the MINIONN network, replace `mnist` with `minionn`)
```bash
cargo build +nightly --release mnist-client --all-features;
cargo build +nightly --release mnist-server --all-features;
```

Then, execute these commands to run the experiment:
```bash
# Start server:
env RAYON_NUM_THREADS=8 cargo +nightly run --release --all-features --bin mnist-server -- -m <0/1> 2>/dev/null > "./mnist.txt"
# Start client:
env RAYON_NUM_THREADS=8 cargo +nightly run --release --all-features --bin mnist-client -- -m <0/1> -i <server_ip> 2>/dev/null > "./mnist.txt"
```
This will write out a trace to `./mnist.txt`.  Note that the pre-processing phase times in this trace will be incorrect.

#### Figures 8 and 9

In the future, MUSE will run Triple Generation in parallel with ACG, Garbling, and Input Authentication. Thus to estimate the latency of the pre-processing phase, compute the result of `max(triple_time, acg_time + garbling_time + input_auth_time) + cds_time`. Once you have calculated that value, you will be able to reconstruct both Figure 8 and 9 using the numbers from before.

## License

Muse is licensed under either of the following licenses, at your discretion.

 * Apache License Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
