import os
import wget
import argparse
import youtokentome as yttm
from tqdm import tqdm


def download_single(directory, fname, url):
    """
    Download a single file from the given URL.

    Args:
        fname (string): Filename it will be saved as.
    """
    full_path = os.path.join(directory, fname)
    if not os.path.exists(full_path):
        print ("Downloading %s ..." % url)
        wget.download(url, full_path)


# https://nlp.stanford.edu/projects/nmt/
def download(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    download_single(directory, 'train_raw.en', "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en")
    download_single(directory, 'train_raw.de', "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de")
    download_single(directory, 'dev.en', "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en")
    download_single(directory, 'dev.de', "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de")
    download_single(directory, 'test.en', "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en")
    download_single(directory, 'test.de', "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de")


def train_bpe(directory, vocab_size):
    with open(os.path.join(directory, 'train_raw.de')) as f:
        de_lines = f.readlines()
        print ("Sentences in 'train_raw.de':", len(de_lines))
    with open(os.path.join(directory, 'train_raw.en')) as f:
        en_lines = f.readlines()
        print ("Sentences in 'train_raw.en':", len(en_lines))

    lines = en_lines + de_lines
    print ("Data size for training BPE model:", len(lines))
    train_data_path = os.path.join(directory, 'train.ende')
    with open(train_data_path, 'w') as f:
        for l in lines:
            f.write(l)
    yttm.BPE.train(data=train_data_path, vocab_size=vocab_size, model='data/bpe.%d.model'%vocab_size,  unk_id=0, pad_id=1, bos_id=2, eos_id=3)
    os.remove(train_data_path)


def clean_train(directory, vocab_size, limit):
    """
    Clean training set by filtering out sentences that are too short or too long.
    """
    print ("Cleaning training set ...")
    with open(os.path.join(directory, 'train_raw.de')) as f:
        de_lines = [l.strip() for l in f.readlines()]
    with open(os.path.join(directory, 'train_raw.en')) as f:
        en_lines = [l.strip() for l in f.readlines()]

    bpe_model = yttm.BPE(model='data/bpe.%d.model'%vocab_size)

    fen = open(os.path.join(directory, 'train.en'), 'w')
    fde = open(os.path.join(directory, 'train.de'), 'w')
    for (x,y) in tqdm(tuple(zip(en_lines, de_lines))):
        if (limit[0] < len(bpe_model.encode(x)) < limit[1]) and (limit[0] < len(bpe_model.encode(y)) < limit[1]):
            fen.write(x+'\n')
            fde.write(y+'\n')
    fen.close()
    fde.close()


def main():
    parser = argparse.ArgumentParser(description="Download datasets and train the BPE model.")
    parser.add_argument('directory', type=str, help="Directory to save datasets.")
    parser.add_argument('--vocab_size', default=32000, type=int,
                        help="Byte-pair encoding (shared source-target) vocabulary size. Default: 37000.")
    parser.add_argument('--limit', type=str, default="1,100", help="Limitation of sequence length (for training set only).")
    args = parser.parse_args()

    # Download
    download(args.directory)
    # Train BPE model
    if not os.path.exists('data/bpe.%d.model'%args.vocab_size):
        train_bpe(args.directory, args.vocab_size)
    # Clean training set
    limit = [int(i) for i in args.limit.split(',')]
    clean_train(args.directory, args.vocab_size, limit)

    print ("Data preparation is complete!")


if __name__ == '__main__':
    main()
