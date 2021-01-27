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

### Instance setup
### Experiments

## License

Muse is licensed under either of the following licenses, at your discretion.

 * Apache License Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
