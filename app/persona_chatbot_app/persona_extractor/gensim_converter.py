import gensim
import argparse
import pickle


def main(filepath: str, outpath: str):
    model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)

    n_dict = {}

    for key in model.wv.vocab:
        n_dict[key] = model[key]

    with open(outpath, mode='wb') as f:
        pickle.dump(n_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gensimのw2vモデルをdictに変換する")

    parser.add_argument("file", help="gensimのw2vファイル")
    parser.add_argument("outpath", help="出力先")

    args = parser.parse_args()

    main(args.file, args.outpath)
