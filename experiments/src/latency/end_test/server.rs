use clap::{App, Arg, ArgMatches};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn get_args() -> ArgMatches<'static> {
    App::new("end-server")
        .arg(
            Arg::with_name("model")
                .short("m")
                .long("model")
                .takes_value(true)
                .help("MNIST (0) MiniONN (1)")
                .required(true),
        )
        .arg(
            Arg::with_name("port")
                .short("p")
                .long("port")
                .takes_value(true)
                .help("Port to listen on (default 8000)")
                .required(false),
        )
        .get_matches()
}

fn main() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let args = get_args();
    let model = clap::value_t!(args.value_of("model"), usize).unwrap();
    assert!(model == 0 || model == 1);
    let num_triples = if model == 0 { 3639680 } else { 60915712 };
    let num_rand = if model == 0 { 909920 } else { 15228928 };

    let port = args.value_of("port").unwrap_or("8000");
    let server_addr = format!("0.0.0.0:{}", port);

    experiments::latency::server::end_test(&server_addr, num_triples, num_rand, &mut rng);
}
