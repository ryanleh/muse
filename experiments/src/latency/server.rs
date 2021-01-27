use crate::*;
use algebra::{
    fields::near_mersenne_64::F,
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::{Fp64, Fp64Parameters},
    FpParameters, PrimeField, UniformRandom,
};
use crypto_primitives::{
    gc::fancy_garbling::{Encoder, GarbledCircuit, Wire},
    AuthAdditiveShare, AuthShare, Share,
};
use neural_network::{
    layers::*,
    tensors::{Input, Output},
    NeuralArchitecture, NeuralNetwork,
};
use num_traits::identities::Zero;
use protocols::{
    gc::ServerGcMsgSend,
    linear_layer::LinearProtocol,
    mpc::{ServerMPC, MPC},
    mpc_offline::{OfflineMPC, ServerOfflineMPC},
    neural_network::NNProtocol,
    server_keygen,
};
use protocols_sys::{server_acg, SealClientACG, SealServerACG, ServerACG, ServerFHE};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;
use scuttlebutt::Block;
use std::{
    collections::BTreeMap,
    io::{BufReader, BufWriter, Write},
    net::TcpListener,
};

pub fn nn_server<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let server_listener = TcpListener::bind(server_addr).unwrap();
    let (server_offline_state, offline_bytes) = {
        let stream = server_listener
            .incoming()
            .next()
            .unwrap()
            .expect("server connection failed!");
        let read_stream = BufReader::new(&stream);
        let mut write_stream = CountWrite::new(BufWriter::new(&stream));
        (
            NNProtocol::offline_server_protocol(read_stream, &mut write_stream, &nn, rng).unwrap(),
            write_stream.count(),
        )
    };

    let (_, online_bytes) = {
        let stream = server_listener
            .incoming()
            .next()
            .unwrap()
            .expect("server connection failed!");
        let read_stream = BufReader::new(&stream);
        let mut write_stream = CountWrite::new(BufWriter::new(&stream));
        (
            NNProtocol::online_server_protocol(read_stream, &mut write_stream, &nn, &server_offline_state).unwrap(),
            write_stream.count()
        )
    };
    add_to_trace!(|| "Offline Phase Bytes written: ", || format!("{}", offline_bytes));
    add_to_trace!(|| "Online Phase Bytes written: ", || format!("{}", online_bytes));
}

pub fn acg<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let server_listener = TcpListener::bind(server_addr).unwrap();
    let stream = server_listener
        .incoming()
        .next()
        .unwrap()
        .expect("Client connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = CountWrite::new(BufWriter::new(stream));

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    let mac_key = F::uniform(rng);

    let mut linear_shares: BTreeMap<
        usize,
        (
            Input<AuthAdditiveShare<F>>,
            Output<F>,
            Output<AuthAdditiveShare<F>>,
        ),
    > = BTreeMap::new();
    let mut mac_keys: BTreeMap<usize, (F, F)> = BTreeMap::new();

    let linear_time = timer_start!(|| "Linear layers offline phase");
    for (i, layer) in nn.layers.iter().enumerate() {
        match layer {
            Layer::NLL(NonLinearLayer::ReLU { dims, .. }) => {}
            Layer::LL(layer) => {
                let (shares, keys) = match &layer {
                    LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                        let mut acg_handler = match &layer {
                            LinearLayer::Conv2d { .. } => SealServerACG::Conv2D(
                                server_acg::Conv2D::new(&sfhe, &layer, &layer.kernel_to_repr()),
                            ),
                            LinearLayer::FullyConnected { .. } => {
                                SealServerACG::FullyConnected(server_acg::FullyConnected::new(
                                    &sfhe,
                                    &layer,
                                    &layer.kernel_to_repr(),
                                ))
                            }
                            _ => unreachable!(),
                        };
                        LinearProtocol::<TenBitExpParams>::offline_server_acg_protocol(
                            &mut reader,
                            &mut writer,
                            layer.input_dimensions(),
                            layer.output_dimensions(),
                            &mut acg_handler,
                            rng,
                        )
                        .unwrap()
                    }
                    LinearLayer::AvgPool { dims, .. } | LinearLayer::Identity { dims } => {
                        let in_zero = Output::zeros(dims.input_dimensions());
                        if linear_shares.keys().any(|k| k == &(i - 1)) {
                            // If the layer comes after a linear layer, apply the function to
                            // the last layer's output share MAC
                            let prev_mac_keys = mac_keys.get(&(i - 1)).unwrap();
                            let prev_output_share = &linear_shares.get(&(i - 1)).unwrap().2;
                            let mut output_share = Output::zeros(dims.output_dimensions());
                            layer.evaluate_naive_auth(prev_output_share, &mut output_share);
                            (
                                (
                                    Input::auth_share_from_parts(in_zero.clone(), in_zero.clone()),
                                    Output::zeros(dims.output_dimensions()),
                                    output_share,
                                ),
                                prev_mac_keys.clone(),
                            )
                        } else {
                            // If the layer comes after a non-linear layer, receive the
                            // randomizer from the client, authenticate it, and then apply the
                            // function to the MAC share
                            let (key, input_share) =
                                LinearProtocol::<TenBitExpParams>::offline_server_auth_share(
                                    &mut reader,
                                    &mut writer,
                                    dims.input_dimensions(),
                                    &sfhe,
                                    rng,
                                )
                                .unwrap();
                            let mut output_share = Output::zeros(dims.output_dimensions());
                            layer.evaluate_naive_auth(&input_share, &mut output_share);
                            (
                                (
                                    -input_share,
                                    Output::zeros(dims.output_dimensions()),
                                    output_share,
                                ),
                                (key, key),
                            )
                        }
                    }
                };
                linear_shares.insert(i, shares);
                mac_keys.insert(i, keys);
            }
        }
    }
    timer_end!(linear_time);
    println!("Bytes written: {}", writer.count());
    add_to_trace!(|| "Bytes written: ", || format!("{}", writer.count()));
}

pub fn garbling<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let server_listener = TcpListener::bind(server_addr).unwrap();
    let stream = server_listener
        .incoming()
        .next()
        .unwrap()
        .expect("Client connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = CountWrite::new(BufWriter::new(stream));

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    let out_mac_keys = vec![F::zero(); layers.len()];
    let out_mac_shares = vec![F::zero(); activations];
    let inp_mac_keys = vec![F::zero(); layers.len()];
    let inp_mac_shares = vec![F::zero(); activations];
    let labels: Vec<(Block, Block)> = vec![(0.into(), 0.into()); 2 * activations * modulus_bits];

    // Generate triples
    let garble_time = timer_start!(|| "Garbling Time");
    let mut gc_s = Vec::with_capacity(activations);
    let mut encoders = Vec::with_capacity(activations);
    let p = (<<F as PrimeField>::Params>::MODULUS.0).into();

    assert_eq!(activations, layers.iter().fold(0, |sum, &x| sum + x));
    let garble_time = timer_start!(|| "Garbling");

    // For each layer, garbled a circuit with the correct number of truncations
    for (i, num) in layers.iter().enumerate() {
        let c = protocols::gc::make_truncated_relu::<TenBitExpParams>(
            TenBitExpParams::EXPONENT_CAPACITY,
        );
        let (en, gc): (Vec<Encoder>, Vec<GarbledCircuit>) = (0..*num)
            .into_par_iter()
            .map(move |_| {
                let mut c = c.clone();
                let (en, gc) = fancy_garbling::garble(&mut c).unwrap();
                (en, gc)
            })
            .unzip();
        encoders.extend(en);
        gc_s.extend(gc);
    }
    timer_end!(garble_time);

    let encode_time = timer_start!(|| "Encoding inputs");
    let (num_garbler_inputs, num_evaluator_inputs) = if activations > 0 {
        (
            encoders[0].num_garbler_inputs(),
            encoders[0].num_evaluator_inputs(),
        )
    } else {
        (0, 0)
    };

    let zero_inputs = vec![0u16; num_evaluator_inputs];
    let one_inputs = vec![1u16; num_evaluator_inputs];
    let mut labels = Vec::with_capacity(activations * num_evaluator_inputs);
    let mut randomizer_labels = Vec::with_capacity(activations);
    let mut output_randomizers = Vec::with_capacity(activations);
    for enc in encoders.iter() {
        // Output server randomization share
        let r = F::uniform(rng);
        output_randomizers.push(r);
        let r_bits: u64 = ((-r).into_repr()).into();
        let r_bits =
            fancy_garbling::util::u128_to_bits(r_bits.into(), crypto_primitives::gc::num_bits(p));
        for w in ((num_garbler_inputs / 2)..num_garbler_inputs)
            .zip(r_bits)
            .map(|(i, r_i)| enc.encode_garbler_input(r_i, i))
        {
            randomizer_labels.push(w);
        }

        let all_zeros = enc.encode_evaluator_inputs(&zero_inputs);
        let all_ones = enc.encode_evaluator_inputs(&one_inputs);
        all_zeros
            .into_iter()
            .zip(all_ones)
            .for_each(|(label_0, label_1)| labels.push((label_0.as_block(), label_1.as_block())));
    }

    // Extract out the zero labels for the carry bits since these aren't
    // used in CDS
    let (carry_labels, input_labels): (Vec<_>, Vec<_>) = labels
        .into_iter()
        .enumerate()
        .partition(|(i, _)| (i + 1) % (F::size_in_bits() + 1) == 0);

    let carry_labels: Vec<Block> = carry_labels
        .into_iter()
        .map(|(_, (zero, _))| zero)
        .collect();
    let input_labels: Vec<(Block, Block)> = input_labels.into_iter().map(|(_, l)| l).collect();
    timer_end!(encode_time);

    let send_gc_time = timer_start!(|| "Sending GCs");
    let randomizer_label_per_relu = if activations == 0 {
        8192
    } else {
        randomizer_labels.len() / activations
    };
    for msg_contents in gc_s
        .chunks(8192)
        .zip(randomizer_labels.chunks(randomizer_label_per_relu * 8192))
    {
        let sent_message = ServerGcMsgSend::new(&msg_contents);
        protocols::bytes::serialize(&mut writer, &sent_message).unwrap();
        writer.flush().unwrap();
    }
    timer_end!(send_gc_time);
    timer_end!(garble_time);
    add_to_trace!(|| "Bytes written: ", || format!("{}", writer.count()));
}

pub fn triples_gen<R: RngCore + CryptoRng>(server_addr: &str, num: usize, rng: &mut R) {
    let server_listener = TcpListener::bind(server_addr).unwrap();
    let stream = server_listener
        .incoming()
        .next()
        .unwrap()
        .expect("Client connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    let mac_key = F::uniform(rng);

    // Generate triples
    let server_gen = ServerOfflineMPC::<F, _>::new(&sfhe, mac_key.into_repr().0);
    let triples = timer_start!(|| "Generating triples");
    server_gen.triples_gen(&mut reader, &mut writer, rng, num);
    timer_end!(triples);
}

use async_std::{prelude::*, task};
use protocols::async_server_keygen;

pub fn async_triples_gen<R: RngCore + CryptoRng>(server_addr: &str, num: usize, rng: &mut R) {
    let (mut reader, mut writer) = task::block_on(async {
        let server_listener = async_std::net::TcpListener::bind(server_addr)
            .await
            .unwrap();
        let stream = server_listener
            .incoming()
            .next()
            .await
            .unwrap()
            .expect("Client connection failed!");
        (
            async_std::io::BufReader::new(stream.clone()),
            CountWrite::new(async_std::io::BufWriter::new(stream)),
        )
    });

    let (sfhe, mac_key) = task::block_on(async {
        (
            async_server_keygen(&mut reader).await.unwrap(),
            F::uniform(rng),
        )
    });

    // Generate triples
    let server_gen = ServerOfflineMPC::<F, _>::new(&sfhe, mac_key.into_repr().0);
    let triples = timer_start!(|| "Generating triples");
    let server_triples = server_gen.async_triples_gen(&mut reader, &mut writer, rng, num);
    timer_end!(triples);
    add_to_trace!(|| "Bytes written: ", || format!("{}", writer.count()));
}

pub fn cds<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let server_listener = TcpListener::bind(server_addr).unwrap();
    let stream = server_listener
        .incoming()
        .next()
        .unwrap()
        .expect("Client connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = CountWrite::new(BufWriter::new(stream));

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    let out_mac_keys = vec![F::zero(); layers.len()];
    let out_mac_shares = vec![F::zero(); activations];
    let inp_mac_keys = vec![F::zero(); layers.len()];
    let inp_mac_shares = vec![F::zero(); activations];
    let labels: Vec<(Block, Block)> = vec![(0.into(), 0.into()); 2 * activations * modulus_bits];

    // Generate triples
    protocols::cds::CDSProtocol::<TenBitExpParams>::server_cds(
        &mut reader,
        &mut writer,
        &sfhe,
        layers,
        &out_mac_keys,
        &out_mac_shares,
        &inp_mac_keys,
        &inp_mac_shares,
        labels.as_slice(),
        rng,
    )
    .unwrap();
    add_to_trace!(|| "Bytes written: ", || format!("{}", writer.count()));
}

pub fn input_auth<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    use protocols::client_keygen;
    use protocols_sys::SealCT;

    let server_listener = TcpListener::bind(server_addr).unwrap();
    let stream = server_listener
        .incoming()
        .next()
        .unwrap()
        .expect("Client connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    let mut cfhe = client_keygen(&mut writer).unwrap();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    let out_mac_keys = vec![F::zero(); layers.len()];
    let out_mac_shares = vec![F::zero(); activations];
    let inp_mac_keys = vec![F::zero(); layers.len()];
    let inp_mac_shares = vec![F::zero(); activations];
    let zero_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];
    let one_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];

    let num_rands = 2 * (activations + activations * modulus_bits);

    // Generate rands
    let mac_key = F::uniform(rng);
    let gen = ServerOfflineMPC::new(&sfhe, mac_key.into_repr().0);

    let input_time = timer_start!(|| "Input Auth");
    let rands = gen.rands_gen(&mut reader, &mut writer, rng, num_rands);
    let mut mpc = ServerMPC::new(rands, Vec::new(), mac_key);

    // Share inputs
    let share_time = timer_start!(|| "Server sharing inputs");
    let out_mac_keys = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_keys.as_slice(), rng)
        .unwrap();
    let inp_mac_keys = mpc
        .private_inputs(&mut reader, &mut writer, inp_mac_keys.as_slice(), rng)
        .unwrap();
    let out_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_shares.as_slice(), rng)
        .unwrap();
    let inp_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, inp_mac_shares.as_slice(), rng)
        .unwrap();
    let zero_labels = mpc
        .private_inputs(&mut reader, &mut writer, zero_labels.as_slice(), rng)
        .unwrap();
    let one_labels = mpc
        .private_inputs(&mut reader, &mut writer, one_labels.as_slice(), rng)
        .unwrap();
    timer_end!(share_time);

    // Receive client shares
    let recv_time = timer_start!(|| "Server receiving inputs");
    let out_bits = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations * modulus_bits)
        .unwrap();
    let inp_bits = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations * modulus_bits)
        .unwrap();
    let c_out_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations)
        .unwrap();
    let c_inp_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations)
        .unwrap();
    timer_end!(recv_time);
    timer_end!(input_time);
}

pub fn async_input_auth<R: RngCore + CryptoRng>(server_addr: &str, server_addr_2: &str, layers: &[usize], rng: &mut R) {
    use protocols::async_client_keygen;
    use protocols_sys::SealCT;
   
    // TODO: Need a sync stream for now because async is not implemented for online MPC
    let server_listener = TcpListener::bind(server_addr).unwrap();
    let stream = server_listener
        .incoming()
        .next()
        .unwrap()
        .expect("Client connection failed!");
    let mut sync_reader = BufReader::new(stream.try_clone().unwrap());
    let mut sync_writer = CountWrite::new(BufWriter::new(stream));

    let (mut reader, mut writer) = task::block_on(async {
        let server_listener = async_std::net::TcpListener::bind(server_addr_2)
            .await
            .unwrap();
        let stream = server_listener
            .incoming()
            .next()
            .await
            .unwrap()
            .expect("Client connection failed!");
        (
            async_std::io::BufReader::new(stream.clone()),
            CountWrite::new(async_std::io::BufWriter::new(stream)),
        )
    });

    let (sfhe, mut cfhe) = task::block_on(async {
        (
            async_server_keygen(&mut reader).await.unwrap(),
            async_client_keygen(&mut writer).await.unwrap(),
        )
    });

    writer.reset();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    let out_mac_keys = vec![F::zero(); layers.len()];
    let out_mac_shares = vec![F::zero(); activations];
    let inp_mac_keys = vec![F::zero(); layers.len()];
    let inp_mac_shares = vec![F::zero(); activations];
    let zero_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];
    let one_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];

    let num_rands = 2 * (activations + activations * modulus_bits);

    // Generate rands
    let mac_key = F::uniform(rng);
    let gen = ServerOfflineMPC::new(&sfhe, mac_key.into_repr().0);

    let input_time = timer_start!(|| "Input Auth");
    let rands = gen.async_rands_gen(&mut reader, &mut writer, rng, num_rands);
    let mut mpc = ServerMPC::new(rands, Vec::new(), mac_key);

    // Share inputs
    let share_time = timer_start!(|| "Server sharing inputs");
    let out_mac_keys = mpc
        .private_inputs(&mut sync_reader, &mut sync_writer, out_mac_keys.as_slice(), rng)
        .unwrap();
    let inp_mac_keys = mpc
        .private_inputs(&mut sync_reader, &mut sync_writer, inp_mac_keys.as_slice(), rng)
        .unwrap();
    let out_mac_shares = mpc
        .private_inputs(&mut sync_reader, &mut sync_writer, out_mac_shares.as_slice(), rng)
        .unwrap();
    let inp_mac_shares = mpc
        .private_inputs(&mut sync_reader, &mut sync_writer, inp_mac_shares.as_slice(), rng)
        .unwrap();
    let zero_labels = mpc
        .private_inputs(&mut sync_reader, &mut sync_writer, zero_labels.as_slice(), rng)
        .unwrap();
    let one_labels = mpc
        .private_inputs(&mut sync_reader, &mut sync_writer, one_labels.as_slice(), rng)
        .unwrap();
    timer_end!(share_time);

    // Receive client shares
    let recv_time = timer_start!(|| "Server receiving inputs");
    let out_bits = mpc
        .recv_private_inputs(&mut sync_reader, &mut sync_writer, activations * modulus_bits)
        .unwrap();
    let inp_bits = mpc
        .recv_private_inputs(&mut sync_reader, &mut sync_writer, activations * modulus_bits)
        .unwrap();
    let c_out_mac_shares = mpc
        .recv_private_inputs(&mut sync_reader, &mut sync_writer, activations)
        .unwrap();
    let c_inp_mac_shares = mpc
        .recv_private_inputs(&mut sync_reader, &mut sync_writer, activations)
        .unwrap();
    timer_end!(recv_time);
    timer_end!(input_time);
    add_to_trace!(|| "Bytes written: ", || format!("{}", writer.count() + sync_writer.count()));
}

pub fn input_auth_ltme<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    use protocols::client_keygen;
    use protocols_sys::SealCT;

    let server_listener = TcpListener::bind(server_addr).unwrap();
    let stream = server_listener
        .incoming()
        .next()
        .unwrap()
        .expect("Client connection failed!");
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = BufWriter::new(stream);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    let mut cfhe = client_keygen(&mut writer).unwrap();

    // Generate and send MAC key to client
    let mac_key = F::uniform(rng);
    let mut mac_ct_seal = SealCT::new();
    let mac_ct = mac_ct_seal.encrypt_vec(&cfhe, vec![mac_key.into_repr().0, 8192]);

    let gen = ServerOfflineMPC::<F, _>::new(&sfhe, mac_key.into_repr().0);
    gen.send_mac(&mut writer, mac_ct);
    let mut mpc = ServerMPC::new(Vec::new(), Vec::new(), mac_key);

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    let out_mac_keys = vec![F::zero(); layers.len()];
    let out_mac_shares = vec![F::zero(); activations];
    let inp_mac_keys = vec![F::zero(); layers.len()];
    let inp_mac_shares = vec![F::zero(); activations];
    let zero_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];
    let one_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];

    let input_time = timer_start!(|| "Input Auth");

    // Share inputs
    let share_time = timer_start!(|| "Server sharing inputs");
    let out_mac_keys = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_keys.as_slice(), rng)
        .unwrap();
    let inp_mac_keys = mpc
        .private_inputs(&mut reader, &mut writer, inp_mac_keys.as_slice(), rng)
        .unwrap();
    let out_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_shares.as_slice(), rng)
        .unwrap();
    let inp_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, inp_mac_shares.as_slice(), rng)
        .unwrap();
    let zero_labels = mpc
        .private_inputs(&mut reader, &mut writer, zero_labels.as_slice(), rng)
        .unwrap();
    let one_labels = mpc
        .private_inputs(&mut reader, &mut writer, one_labels.as_slice(), rng)
        .unwrap();
    timer_end!(share_time);

    // Receive client shares
    let recv_time = timer_start!(|| "Server receiving inputs");
    let out_bits = gen.recv_optimized_input(&cfhe, &mut reader, activations * modulus_bits);
    let inp_bits = gen.recv_optimized_input(&cfhe, &mut reader, activations * modulus_bits);
    let c_out_mac_shares = gen.recv_optimized_input(&cfhe, &mut reader, activations);
    let c_inp_mac_shares = gen.recv_optimized_input(&cfhe, &mut reader, activations);
    timer_end!(recv_time);
    timer_end!(input_time);
}
