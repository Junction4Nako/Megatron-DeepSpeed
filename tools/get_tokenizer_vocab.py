import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='save the pre-trained tokenizer to specified path')
    parser.add_argument('--tokenizer_name', type=str, default=None, help='the tokenizer name to load')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save, only for vocab.json + merges.txt')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.save_vocabulary(args.save_path)

if __name__=='__main__':
    main()
