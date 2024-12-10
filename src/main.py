# src/main.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
from train import main as train_main
from evaluate import main as evaluate_main
from generate_submission import main as generate_submission_main
os.environ["HF_HUB_OFFLINE"] = "False"

def main():
    parser = argparse.ArgumentParser(description='Plant Pathology EfficientNet Training Pipeline')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'submit'],
                        help='Mode to run: train, evaluate, submit')
    args = parser.parse_args()

    if args.mode == 'train':
        train_main()
    elif args.mode == 'evaluate':
        evaluate_main()
    elif args.mode == 'submit':
        generate_submission_main()
    else:
        print("无效的模式选择。请选择 train, evaluate 或 submit。")

if __name__ == '__main__':
    main()
