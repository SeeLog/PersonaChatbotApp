from flask import Flask, request, jsonify

import json

import torch
from persona_chatbot_app.chatbot import TargetPersonaChatBot
from persona_chatbot_app.model.transformer import utils
from persona_chatbot_app.model.transformer.models import make_target_persona_model
from persona_chatbot_app.options import SentencePieceSimpleOption
from persona_chatbot_app.data.tokenizer.sentence_piece import SentencepieceTokenizer
from persona_chatbot_app.persona_extractor.persona_extractor import PersonaExtractor
from persona_chatbot_app.data.tokenizer.mecab_end_concat import MecabEndConcatTokenizer

import pickle
from typing import Optional

import argparse


def _get_persona(sentence: str) -> Optional[torch.tensor]:
    persona, words = extractor.extract(sentence)

    if persona is not None:
        return torch.tensor(persona).to(device).float().repeat(1, 1, opt.max_length).view(1, opt.max_length, -1), words
    else:
        return None, None



parser = argparse.ArgumentParser(description='サーバー起動オプションを指定できます')

parser.add_argument('-s', '--sentencepiece', default='./data/sentencepiece.model', help='SentencePieceのモデルを指定します．よくわからなければデフォルトでOKです．')
parser.add_argument('-v', '--vocab', default='./data/vocab.bin', help='Chatbotのモデル用のVocabファイルを指定します．SentencePieceのものとは別です．')
parser.add_argument('-m', '--model', default='./data/checkpoints/best_model.pt', help='モデルの重みファイルを指定します．')
parser.add_argument('-w', '--word_vec', default='./data/style_sensitive_dict.bin', help='Style-sensitive word vectorsの学習済みモデルを指定します．')
parser.add_argument('-f', '--first_persona', default='やるでやんす', help='最初にセットするペルソナを指定します．')
parser.add_argument('-d', '--device', default='', help='モデルを利用するデバイスを指定します．空の場合，自動で選択をします．')

parser.add_argument('-p', '--port', default=5000, type=int, help='ポート番号を指定します．')
parser.add_argument('-ip', '--ip', default='0.0.0.0', help='IPもしくはホストを指定します．')


args = parser.parse_args()


opt = SentencePieceSimpleOption()

# with open(args.vocab, mode='rb') as f:
#     vocab = cloudpickle.load(f)

with open(args.vocab, mode='rb') as f:
    vocab = pickle.load(f)

tokenizer = SentencepieceTokenizer(args.sentencepiece)
# fields = JapaneseTextWithID(tokenizer, max_length=opt.max_length)

# fields.src.vocab = vocab
# fields.tgt.vocab = vocab

if args.device == '':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = args.device

print("使用デバイス:", device)

print("モデルを読み込みます")
model = make_target_persona_model(len(vocab), -1, opt.num_layer, opt.embedding_size, opt.hidden_size, opt.num_head, opt.dropout)

model = utils.load_checkpoint(model, args.model, device)
model.eval()
print("モデル読み込みOK")

# chatbot = TargetPersonaChatBot(model, fields, device)
chatbot = TargetPersonaChatBot(model, tokenizer, vocab, device)


# ペルソナ用のトークナイザはMeCabベースのものにする
extractor = PersonaExtractor(args.word_vec, tokenizer=MecabEndConcatTokenizer(), verbose=True)


api = Flask(__name__)

response = "hello"

current_persona, current_persona_words = _get_persona(args.first_persona)

# 最後にセットされたのが圧縮されたペルソナかどうかのフラグ
last_minimized = False


def json_decode():
    data = request.data.decode('utf-8')
    data = json.loads(data)

    return data


def minimize_tensor(tensor: torch.tensor) -> torch.tensor:
    """
    与えられたペルソナテンソルを圧縮する
    """
    with torch.no_grad():
        encoded = model.auto_encoder.encode(tensor)

        return encoded


def to_vector(tensor: torch.tensor) -> torch.tensor:
    """
    最大長分だけ拡張されたペルソナテンソルをペルソナベクトルに変換する
    e.g. [1, 64, 16] -> [16]
    """
    return tensor[0, 0, :]


@api.route("/generateReply", methods=["POST"])
def generate_reply():
    """
    JSONでリプライを返す
    """
    global current_persona
    # sentence = request.args.get("sentence")
    json = json_decode()

    sentence = json.get("sentence")

    if sentence is not None:
        reply = chatbot(sentence, current_persona, from_minimized_persona=last_minimized).replace(" ", "").replace("▁", " ")
        return jsonify({"reply": reply})
    else:
        return jsonify({"reply": None})


@api.route("/setPersona", methods=["POST"])
def set_persona():
    """
    渡された文からペルソナを計算してセットする
    ついでに計算結果を返すかどうかを is_return_vector で指定もできる
    """
    global current_persona
    global current_persona_words
    global last_minimized

    json = json_decode()

    # sentence = request.form.get("sentence")
    sentence = json.get("sentence")

    persona_vec = json.get("persona_vector")

    # return_tensor = request.form.get("is_return_tensor")
    return_tensor = json.get("is_return_vector")

    if return_tensor is None:
        return_tensor = False

    if sentence is not None:
        persona, words = extractor.extract(sentence)

        if persona is not None:
            current_persona = torch.tensor(persona).to(device).float().repeat(1, 1, opt.max_length).view(1, opt.max_length, -1)
            current_persona_words = words

            last_minimized = False

            ret = {"success": True, "words": words}
            if return_tensor:
                ret["persona_vector"] = to_vector(minimize_tensor(current_persona)).tolist()

            return jsonify(ret)
        else:
            ret = {"success": False, "words": None, "persona_vector": None}

            if return_tensor:
                ret["persona_vector"] = None

            return jsonify(ret)
    elif persona_vec is not None:
        persona_tensor = torch.tensor(persona_vec).to(device).float().repeat(1, 1, opt.max_length).view(1, opt.max_length, -1)

        last_minimized = True

        ret = {"success": True, "words": None}

        current_persona = persona_tensor
        current_persona_words = []

        if return_tensor:
            ret["persona_vector"] = to_vector(persona_tensor).tolist()

        return jsonify(ret)
    else:
        return jsonify({"success": False})


@api.route("/getPersona", methods=["POST"])
def get_persona():
    """
    渡された文からペルソナを計算して結果を返す
    計算するだけで反映はさせない
    """
    # sentence = request.args.get["sentence"]
    json = json_decode()
    sentence = json.get("sentence")

    if sentence is not None:
        persona_tensor, words = _get_persona(sentence)

        return jsonify({"words": words, "persona_vector": to_vector(minimize_tensor(persona_tensor)).tolist()})
    else:
        return jsonify({"persona_vector": None, "words": None})


@api.route("/getCurrentPersona", methods=["GET"])
def get_current_persona():
    """
    現在セットされているペルソナを返します
    """
    ret = {}

    if last_minimized:
        ret["persona_vector"] = to_vector(current_persona).tolist()
    else:
        ret["persona_vector"] = to_vector(minimize_tensor(current_persona)).tolist()

    return jsonify(ret)


def main():
    print("サーバを起動します\n\n\n")
    api.run(host=args.ip, port=args.port)


if __name__ == '__main__':
    main()
