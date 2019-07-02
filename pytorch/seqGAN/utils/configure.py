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
    parser.add_argument('--pre_gen_max_epoch', type=int, default=100)
    parser.add_argument('--pre_dis_max_epoch', type=int, default=100)
    parser.add_argument('--max_epoch', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--adam_lr', type=float, default=2e-4, help='lr in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.5, help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='beta2 in Adam optimizer')

    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--seq_len_en', type=int, default=20)
    parser.add_argument('--seq_len_jp', type=int, default=25)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    args = parser.parse_args()

    return args