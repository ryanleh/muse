use crate::*;
use ::neural_network::{tensors::Input, NeuralArchitecture};
use algebra::{fields::near_mersenne_64::F, Field, PrimeField};
use num_traits::identities::Zero;
use protocols::{
    async_client_keygen, client_keygen,
    mpc::{ClientMPC, MPC},
    mpc_offline::{ClientOfflineMPC, OfflineMPC},
    neural_network::NNProtocol,
};
use std::{
    io::{BufReader, BufWriter},
    net::TcpStream,
};

pub fn nn_client<R: RngCore + CryptoRng>(
    num_clients: usize,
    server_addr: &str,
    architectures: &[(
        (usize, usize, usize, usize),
        NeuralArchitecture<TenBitAS, TenBitExpFP>,
    )],
    rng: &mut R,
) {
    // Sample a random input.
    let input_dims = architectures[0]
        .1
        .layers
        .first()
        .unwrap()
        .input_dimensions();
    let batch_size = input_dims.0;

    let mut client_states = Vec::new();
    for (_, architecture) in architectures {
        let client_state = {
            // client's connection to server.
            let stream = TcpStream::connect(server_addr).expect("connecting to server failed");
            let read_stream = BufReader::new(&stream);
            let write_stream = BufWriter::new(&stream);

            NNProtocol::offline_client_protocol(read_stream, write_stream, &architecture, rng)
                .unwrap()
        };
        client_states.push(client_state);
    }

    let mut inputs = Vec::new();
    for (input_dims, _) in architectures {
        let mut input = Input::zeros(*input_dims);
        input
            .iter_mut()
            .for_each(|in_i| *in_i = generate_random_number(rng).1);
        inputs.push(input);
    }

    let start = std::time::Instant::now();
    let _ = crossbeam::thread::scope(|s| {
        let mut results = Vec::new();
        for _ in 0..num_clients {
            let result = s.spawn(|_| {
                let architecture = &architectures[0].1;
                let input = &inputs[0];
                let client_state = &client_states[0];
                let start = std::time::Instant::now();
                let stream = TcpStream::connect(server_addr).expect("connecting to server failed");
                let read_stream = BufReader::new(&stream);
                let write_stream = BufWriter::new(&stream);
                let _ = NNProtocol::online_client_protocol(
                    read_stream,
                    write_stream,
                    &input,
                    architecture,
                    &client_state,
                )
                .unwrap();
                start.elapsed()
            });
            results.push(result);
        }
        for result in results {
            println!(
                "Served {} clients in {}s",
                batch_size,
                result.join().unwrap().as_millis() as f64 / 1000.0
            );
        }
    })
    .unwrap();
    let end = start.elapsed();
    println!(
        "Served {} clients in {}s",
        batch_size * num_clients,
        end.as_millis() as f64 / 1000.0
    );
}
