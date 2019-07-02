import os
import argparse

script_path = os.path.dirname(__file__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, default="/result")
    parser.add_argument('--noise_dist', type=str, default="normal")
    parser.add_argument('--use_lang', type=str, default="jp", choices=["en","jp"])
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--silent', action="store_true")

    parser.add_argument('--display_interval', type=int, default=10, help='Interval of displaying log to console')
    parser.add_argument('--snapshot_interval', type=int, default=10, help='Interval of displaying log to console')
    parser.add_argument('--n_dis', type=int, default=1, help='number of discriminator update per generator update')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--epoch_decay', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--adam_lr', type=float, default=2e-4, help='lr in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.5, help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='beta2 in Adam optimizer')

    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--noise_dim', type=int, default=256)
    parser.add_argument('--sent_dim', type=int, default=1024)
    parser.add_argument('--n_resblock', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--kl_coef', type=float, default=4)
    parser.add_argument('--side_output_at', type=list, default=[64, 128, 256])
    
    args = parser.parse_args()

    return args