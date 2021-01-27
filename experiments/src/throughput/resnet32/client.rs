use experiments::resnet32::construct_resnet_32;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4,
    0x76, 0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a,
    0x52, 0xd2,
];



fn main() {
    let num_polys = std::env::args().nth(1).expect("Please pass number of polynomial layers as input.").parse().expect("should be positive integer"); // Get number of polys
    let num_clients = std::env::args().nth(2).expect("Please pass number of clients as input.").parse().expect("should be positive integer"); // Get number of polys
    let vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());

    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let network = construct_resnet_32(Some(&vs.root()), 10, num_polys, &mut rng);
    let architecture = (&network).into();

    let server_addr = "169.229.49.175:8002";
    experiments::throughput::client::nn_client(num_clients, server_addr, architecture, &mut rng);
}
