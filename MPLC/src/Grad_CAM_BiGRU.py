import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.backend import expand_dims
import keras
from tqdm import tqdm


def fasta2dict(fasta_name):
    fa_dict = {}
    keys = []
    with open(fasta_name) as fa:
        for line in fa:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                seq_name = line[1:]
                fa_dict[seq_name] = ''
                keys.append(seq_name)
            else:
                fa_dict[keys[-1]] += line
    return fa_dict


def iter_feature_table(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feat_other = line.split('\t')[1:]
            yield np.array([list(map(float, feat_other))], dtype=np.float32)


def build_gradcam_function(model, layer_name):
    target_layer = model.get_layer(layer_name)
    layer_output = target_layer.output

    model_output = model.output
    if isinstance(model_output, (list, tuple)):
        model_output = model_output[0]

    pred_index = tf.argmax(model_output[0])
    class_score = tf.gather(model_output[0], pred_index)

    grads = K.gradients(class_score, layer_output)
    grads = grads[0]

    pooled_grads = K.mean(grads, axis=(0, 1))

    iterate = K.function([model.input], [pooled_grads, layer_output[0]])
    return iterate


def compute_heatmap(iterate, X):
    pooled_grads_val, conv_out_val = iterate([X])
    weighted = conv_out_val * pooled_grads_val
    heatmap = np.mean(weighted, axis=-1)
    return np.squeeze(heatmap)


def write_results(fa_dict, mean_heatmap, res_path):
    os.makedirs(res_path, exist_ok=True)
    n = mean_heatmap.shape[0]

    for key, seq in fa_dict.items():
        head = key.strip().split(' ')[0]
        if head.startswith('s'):
            name = head.split('|')[1]
        else:
            name = head
        out_path = os.path.join(res_path, name + ".csv")
        with open(out_path, "w") as f:
            f.write("No.,residue,weight\n")
            for i, aa in enumerate(seq):
                w = float(mean_heatmap[i]) if i < n else 0.0
                f.write(f"{i+1},{aa},{w}\n")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM-BiGRU Algorithm")
    parser.add_argument("-iX", "--input_X", required=True, type=str)
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("-s", "--ss8_file", required=True, type=str)
    parser.add_argument("-r", "--res_path", required=True, type=str)
    parser.add_argument("-l", "--layer", default="bidirectional_1", type=str)
    args = parser.parse_args()

    model = load_model(
        args.model_path,
        custom_objects={
            'keras.backend': keras.backend,
            'expand_dims': expand_dims
        },
    )

    print("model.output:", model.output)

    iterate = build_gradcam_function(model, args.layer)

    fa_dict = fasta2dict(args.ss8_file)
    total_samples = 0
    with open(args.input_X) as f:
        for line in f:
            if line.strip():
                total_samples += 1

    mean_heatmap = None
    for X in tqdm(iter_feature_table(args.input_X),
                  total=total_samples,
                  desc="Computing Grad-CAM-BiGRU heatmaps"):
        hm = compute_heatmap(iterate, X)
        if mean_heatmap is None:
            mean_heatmap = np.zeros_like(hm, dtype=np.float64)
        mean_heatmap += hm

    if total_samples == 0:
        raise ValueError("No samples found in input_X")

    mean_heatmap /= total_samples
    mean_heatmap = np.maximum(mean_heatmap, 0.0)
    mean_heatmap = mean_heatmap / (mean_heatmap.max() + 1e-10)

    print("Final heatmap shape:", mean_heatmap.shape)
    print(mean_heatmap)

    write_results(fa_dict, mean_heatmap, args.res_path)


if __name__ == '__main__':
    main()
