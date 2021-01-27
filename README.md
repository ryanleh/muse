<h1 align="center">Muse</h1>

___Muse___ is a Python, C++, and Rust library for **Secure Convolutional Neural Network Inference for Malicious Clients**. 

**WARNING:** This is an academic proof-of-concept prototype, and in particular has not received careful code review. This implementation is NOT ready for production use.

## Overview

This library implements the components of a cryptographic system for efficient client-malicious inference on general convolutional neural networks.

**The end-to-end protocol is not yet implemented**. In particular, the following components are missing:
* TopGear zero-knowledge proof integration (latency costs are currently simulated)
* The various components do not currently run in parallel

The rest of this README will walk through running experiments for the various components of Muse to reproduce the results in Tables 3/4 and Figures 8, 9, and 10.


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

The ID of this AMI is XYZ.

To set up an EC2 machine with our AMI, follow the instructions outlined [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/launching-instance.html), and, in Step 1, select "Community AMI".

To replicate our results from the paper, both the client and the server must be XYZ instances, and the region for the client machine should be XYZ, while the region for the server should be XYZ.

### Experiments

First, `cd rust/experiments` to switch to the crate containing our benchmarks.

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
env RAYON_NUM_THREADS=1 cargo run --release --all-features acg-server -- -m <0/1> -p <port> > "./acg_times.csv" &;
# Start client:
env RAYON_NUM_THREADS=1 cargo run --release --all-features acg-client -- -m <0/1> -i <server_ip> -p <server_port> > /dev/null 2>&1;
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
env RAYON_NUM_THREADS=1 cargo run --release --all-features garbling-server -- -m <0/1> -p <port> > "./garbling_times.csv" &;
# Start client:
env RAYON_NUM_THREADS=1 cargo run --release --all-features garbling-client -- -m <0/1> -i <server_ip> -p <server_port> > /dev/null 2>&1;
```
This will write out a trace of execution times to  `./garbling_times.csv`
To obtain the final time, run `cat garbling_times.csv | grep "Garbling"`

##### CDS

To measure the cost of evaluating the CDS protocol, first build the relevant binaries:
```bash
cargo build --release cds-client --all-features;
cargo build --release cds-server --all-features;
```

Then, in separate windows, execute these binaries with the relevant options:
```bash
# Start server:
env RAYON_NUM_THREADS=1 cargo run --release --all-features cds-server -- -m <0/1> -p <port> > "./cds_times.csv" &;
# Start client:
env RAYON_NUM_THREADS=1 cargo run --release --all-features cds-client -- -m <0/1> -i <server_ip> -p <server_port> > /dev/null 2>&1;
```
This will write out a trace of execution times to  `./cds_times.csv`
To obtain the final time, run `cat cds_times.csv | grep "CDS Protocol"`

To measure the cost of triple generation for the CDS protocol, first build the relevant binaries:
```bash
cargo build --release triples-gen-client --all-features;
cargo build --release triples-gen-server --all-features;
```

Then, in separate windows, execute these binaries with the relevant options:
```bash
# Start server:
env RAYON_NUM_THREADS=1 cargo run --release --all-features triples-gen-server -- -m <0/1> -p <port> > "./triple_times.csv" &;
# Start client:
env RAYON_NUM_THREADS=1 cargo run --release --all-features triples-gen-client -- -m <0/1> -i <server_ip> -p <server_port> > /dev/null 2>&1;
```
This will write out a trace to `./triples_times.csv`

To measure the cost of input sharing for the CDS protocol, first build the relevant binaries:
```bash
cargo build --release input-auth-client --all-features;
cargo build --release input-auth-server --all-features;
```

Then, in separate windows, execute these binaries with the relevant options:
```bash
# Start server:
env RAYON_NUM_THREADS=1 cargo run --release --all-features input-auth-server -- -m <0/1> -p <port> > "./input_auth_times.csv" &;
# Start client:
env RAYON_NUM_THREADS=1 cargo run --release --all-features input-auth-client -- -m <0/1> -i <server_ip> -p <server_port> > /dev/null 2>&1;
```
This will write out a trace to `./input_auth_times.csv`

##### Online phase

To measure the cost of input sharing for the CDS protocol, first build the relevant binaries.
(The code examples show how to do this for MNIST; for the MINIONN network, replace `mnist` with `minionn`)
```bash
cargo build --release mnist-client --all-features;
cargo build --release mnist-server --all-features;
```

Then, in separate windows, execute these binaries with the relevant options:
```bash
# Start server:
env RAYON_NUM_THREADS=1 cargo run --release --all-features mnist-server -- -m <0/1> -p <port> > "./mnist.csv" &;
# Start client:
env RAYON_NUM_THREADS=1 cargo run --release --all-features mnist-client -- -m <0/1> -i <server_ip> -p <server_port> > /dev/null 2>&1;
```
This will write out a trace to `./mnist.csv`
## License

Muse is licensed under either of the following licenses, at your discretion.

 * Apache License Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
