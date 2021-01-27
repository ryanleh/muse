use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

// It may be the case that down-sampling happens here.
fn conv_block<R: RngCore + CryptoRng>(
    nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    (k_h, k_w): (usize, usize),
    num_output_channels: usize,
    stride: usize,
    rng: &mut R,
) {
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let c_in = cur_input_dims.1;

    let (conv_1, _) = sample_conv_layer(
        vs,
        cur_input_dims,
        (num_output_channels, c_in, k_h, k_w),
        stride,
        Padding::Same,
        rng,
    );
    nn.layers.push(Layer::LL(conv_1));
    add_activation_layer(nn);
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let c_in = cur_input_dims.1;

    let (conv_2, _) = sample_conv_layer(
        vs,
        cur_input_dims,
        (c_in, c_in, k_h, k_w), // Kernel dims
        1,                      // Stride = 1
        Padding::Same,
        rng,
    );
    nn.layers.push(Layer::LL(conv_2));
    add_activation_layer(nn);
}

// There's no down-sampling happening here, strides are always (1, 1).
fn iden_block<R: RngCore + CryptoRng>(
    nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    (k_h, k_w): (usize, usize),
    rng: &mut R,
) {
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let c_in = cur_input_dims.1;

    let (conv_1, _) = sample_conv_layer(
        vs,
        cur_input_dims,
        (c_in, c_in, k_h, k_w), // Kernel dims
        1,                      // stride
        Padding::Same,
        rng,
    );
    nn.layers.push(Layer::LL(conv_1));
    add_activation_layer(nn);

    let (conv_2, _) = sample_conv_layer(
        vs,
        cur_input_dims,
        (c_in, c_in, k_h, k_w), // Kernel dims
        1,                      // stride
        Padding::Same,
        rng,
    );
    nn.layers.push(Layer::LL(conv_2));
    add_activation_layer(nn);
}

fn resnet_block<R: RngCore + CryptoRng>(
    nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>,
    vs: Option<&tch::nn::Path>,
    layer_size: usize,
    c_out: usize,
    kernel_size: (usize, usize),
    stride: usize,
    rng: &mut R,
) {
    conv_block(nn, vs, kernel_size, c_out, stride, rng);
    for _ in 0..(layer_size - 1) {
        iden_block(nn, vs, kernel_size, rng)
    }
}

pub fn construct_resnet_32<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    // Dimensions of input image.
    let input_dims = (batch_size, 3, 32, 32);
    // Dimensions of first kernel
    let kernel_dims = (16, 3, 3, 3);

    // Sample a random kernel.
    let (conv_1, _) = sample_conv_layer(
        vs,
        input_dims,
        kernel_dims,
        1, // Stride
        Padding::Same,
        rng,
    );
    network.layers.push(Layer::LL(conv_1));
    add_activation_layer(&mut network);
    resnet_block(
        &mut network,
        vs,
        5,      // layer_size,
        16,     // c_out
        (3, 3), // kernel_size
        1,      // stride
        rng,
    );

    resnet_block(
        &mut network,
        vs,
        5,      // layer_size,
        32,     // c_out
        (3, 3), // kernel_size
        2,      // stride
        rng,
    );

    resnet_block(
        &mut network,
        vs,
        5,      // layer_size,
        64,     // c_out
        (3, 3), // kernel_size
        2,      // stride
        rng,
    );
    let avg_pool_input_dims = network.layers.last().unwrap().output_dimensions();
    network.layers.push(Layer::LL(sample_avg_pool_layer(
        avg_pool_input_dims,
        (2, 2),
        2,
    )));

    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    network.layers.push(Layer::LL(fc));
    assert!(network.validate());

    // println!("NLLs:");
    // for layer in &network.layers {
    //     if let Layer::NLL(l) = layer {
    //         println!("layer input: {:?}", layer.input_dimensions());
    //     }
    // }

    // println!("Convs:");
    // for layer in &network.layers {
    //     if let Layer::LL(LinearLayer::Conv2d { dims, params }) = layer {
    //         println!("Layer input dims: {:?}", layer.input_dimensions());
    //         println!("Kernel dims: {:?}", params.kernel.dim());
    //         println!();
    //     }
    // }

    network
}
