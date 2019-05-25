import os
import argparse

script_path = os.path.dirname(__file__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="../../../../data_hdd/uec_data")
    parser.add_argument('--save_path', type=str, default="/result")
    parser.add_argument('--noise_dist', type=str, default="normal")
    parser.add_argument('--name', type=str, default="flower", choices=["MSCOCO","flower","bird"])
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--silent', action="store_true")

    parser.add_argument('--snapshot_interval', type=int, default=10000, help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10, help='Interval of displaying log to console')
    
    parser.add_argument('--max_iter', type=int, default=1000000)
    parser.add_argument('--max_epoch', type=int, default=600)
    parser.add_argument('--n_dis', type=int, default=1, help='number of discriminator update per generator update')

    parser.add_argument("--image_size",action="store",type=int,default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--adam_alpha', type=float, default=0.0002, help='alpha in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 in Adam optimizer')

    parser.add_argument('--num_resblock', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--hid_dim', type=int, default=512)
    parser.add_argument('--sent_dim', type=int, default=128)
    parser.add_argument('--noise_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.75)
    
    args = parser.parse_args()
    args.image_size = tuple((args.image_size,args.image_size))

    return args