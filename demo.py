import argparse
import tkinter as tk

import torch

from isegm.inference import utils
from isegm_gui.app import InteractiveSegmentationGUI


def main():
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    # checkpoint_path = utils.find_checkpoint("weights", args.checkpoint)
    model = utils.load_is_model(args.checkpoint, args.device, cpu_dist_maps=True)

    root = tk.Tk()
    root.minsize(960, 480)
    print(args)
    def callback():
        print("callback")
    app = InteractiveSegmentationGUI(root, model, args.device, callback=callback)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    args = parser.parse_args()
    if args.cpu:
        args.device =torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    # cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args


if __name__ == '__main__':
    main()
