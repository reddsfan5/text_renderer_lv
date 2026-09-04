"""Call-number detection dataset pipeline (LabelMe format)."""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from text_renderer.renderers import callnumber as renderer


def main(arg_list=None):
    parser = argparse.ArgumentParser(
        prog="text-renderer callnumber-detection", description=__doc__
    )
    parser.add_argument("--out", default="./call_number_det_equal_width_1w")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--xlsx", default=renderer.DEFAULT_XLSX, help="path to call-number Excel file"
    )
    parser.add_argument(
        "--include-image-data",
        action="store_true",
        help="embed base64 imageData into LabelMe JSON",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output directory"
    )
    args = parser.parse_args(arg_list)

    if args.num <= 0:
        parser.error("require --num > 0")
    if args.width <= 0 or args.height <= 0:
        parser.error("require --width > 0 and --height > 0")

    output_path = Path(args.out)
    if output_path.exists() and not args.overwrite:
        parser.error(
            "output already exists: {}; pass --overwrite to replace it".format(
                output_path
            )
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    renderer.load_callnumber_pools(args.xlsx)
    renderer._IMAGE_SCALE_FACTOR = args.height / renderer._REFERENCE_IMAGE_HEIGHT
    print(
        "Scale factor: {:.2f} (H={}/BASE={})".format(
            renderer._IMAGE_SCALE_FACTOR,
            args.height,
            renderer._REFERENCE_IMAGE_HEIGHT,
        )
    )

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    counts = []
    for index in tqdm(range(args.num)):
        counts.append(
            renderer.generate_one(
                output_path,
                index,
                W=args.width,
                H=args.height,
                include_imageData=args.include_image_data,
            )
        )

    print("Done: {}".format(output_path))
    print("images: {}".format(args.num))
    print("avg labels/image: {:.2f}".format(sum(counts) / len(counts)))
    return output_path


if __name__ == "__main__":
    main()
