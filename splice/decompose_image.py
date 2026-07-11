import argparse
import sys
import torch
from PIL import Image
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import splice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str)
    parser.add_argument('-out_path', type=str)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('-l1_penalty', type=float, default=0.25)
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-model', type=str, default="open_clip:ViT-B-32")
    parser.add_argument('-vocab', type=str, default="laion")
    parser.add_argument('-vocab_size', type=int, default=10000)
    args = parser.parse_args()

    splicemodel = splice.load(args.model, args.vocab, args.vocab_size, args.device, l1_penalty = args.l1_penalty, return_weights=True)
    preprocess = splice.get_preprocess(args.model)
    img = preprocess(Image.open(args.path)).to(args.device).unsqueeze(0)

    weights, l0_norm, cosine = splice.decompose_image(img, splicemodel, args.device)

    vocab = splice.get_vocabulary(args.vocab, args.vocab_size)

    _, indices = torch.sort(weights, descending=True)

    requested_output = Path(args.out_path)
    if requested_output.suffix:
        outpath = requested_output
    else:
        outpath = requested_output / f"{Path(args.path).stem}_weights.txt"
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with open(outpath, "w") as f:

        f.write("Concept Decomposition of " + str(args.path) + ": \n")
        print("Concept Decomposition of " + str(args.path) + ":")

        for idx in indices.squeeze():
            if weights[0, idx.item()].item() == 0:
                break
            f.write("\t" + str(vocab[idx.item()]) + "\t" + str(round(weights[0, idx.item()].item(), 4)) + "\n")
            if args.verbose:
                print("\t" + str(vocab[idx.item()]) + "\t" + str(round(weights[0, idx.item()].item(), 4)))

        f.write("Decomposition L0 Norm: \t" + str(l0_norm) + "\n")
        print("Decomposition L0 Norm: \t" + str(l0_norm))

        f.write("CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)) + "\n")
        print("CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)))

if __name__ == "__main__":
    main()
