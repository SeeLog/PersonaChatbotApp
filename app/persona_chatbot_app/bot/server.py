from flask import Flask, request


import torch
from persona_chatbot_app.chatbot import TargetPersonaChatBot
from persona_chatbot_app.model.transformer import utils
from persona_chatbot_app.model.transformer.models import make_target_persona_model
from persona_chatbot_app.options import SentencePieceSimpleOption
from persona_chatbot_app.data.tokenizer.sentence_piece import SentencepieceTokenizer
from persona_chatbot_app.data.japanese_text import JapaneseTextWithID
from persona_chatbot_app.persona_extractor.persona_extractor import PersonaExtractor
from persona_chatbot_app.data.tokenizer.mecab_end_concat import MecabEndConcatTokenizer

import cloudpickle
from typing import Optional

import argparse


def _get_persona(sentence: str) -> Optional[torch.tensor]:
    persona, words = extractor.extract(sentence)

    if persona is not None:
        return torch.tensor(persona).to(device).float().repeat(1, 1, opt.max_length).view(1, opt.max_length, -1)
    else:
        return None



parser = argparse.ArgumentParser(description='サーバー起動オプションを指定できます')

parser.add_argument('-s', '--sentencepiece', default='./data/sentencepiece.model', help='SentencePieceのモデルを指定します．よくわからなければデフォルトでOKです．')
parser.add_argument('-v', '--vocab', default='./data/vocab.bin', help='Chatbotのモデル用のVocabファイルを指定します．SentencePieceのものとは別です．')
parser.add_argument('-m', '--model', default='./data/checkpoints/best_model.pt', help='モデルの重みファイルを指定します．')
parser.add_argument('-w', '--word_vec', default='./data/style_sensitive.bin', help='Style-sensitive word vectorsの学習済みモデルを指定します．')
parser.add_argument('-f', '--first_persona', default='やるでやんす', help='最初にセットするペルソナを指定します．')
parser.add_argument('-d', '--device', default='', help='モデルを利用するデバイスを指定します．空の場合，自動で選択をします．')

parser.add_argument('-p', '--port', default=5000, type=int, help='ポート番号を指定します．')
parser.add_argument('-ip', '--ip', default='localhost', help='IPもしくはホストを指定します．')


args = parser.parse_args()


opt = SentencePieceSimpleOption()

with open(args.vocab, mode='rb') as f:
    vocab = cloudpickle.load(f)

tokenizer = SentencepieceTokenizer(args.sentencepiece)
fields = JapaneseTextWithID(tokenizer, max_length=opt.max_length)

fields.src.vocab = vocab
fields.tgt.vocab = vocab

if args.device == '':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = args.device

print("使用デバイス:", device)

print("モデルを読み込むよ")
model = make_target_persona_model(len(vocab), -1, opt.num_layer, opt.embedding_size, opt.hidden_size, opt.num_head, opt.dropout)

model = utils.load_checkpoint(model, args.model, device)
print("モデル読み込みOK")

chatbot = TargetPersonaChatBot(model, fields, device)

# ペルソナ用のトークナイザはMeCabベースのものにする
extractor = PersonaExtractor(args.word_vec, tokenizer=MecabEndConcatTokenizer(), verbose=True)


api = Flask(__name__)

response = "hello"

current_persona = _get_persona(args.first_persona)


@api.route("/generateReply", methods=["POST"])
def generate_reply():
    global current_persona
    sentence = request.form["sentence"]
    return chatbot(sentence, current_persona).replace(" ", "").replace("▁", " ")


@api.route("/setPersona", methods=["POST"])
def set_persona():
    global current_persona
    sentence = request.form["sentence"]
    persona, words = extractor.extract(sentence)

    if persona is not None:
        current_persona = torch.tensor(persona).to(device).float().repeat(1, 1, opt.max_length).view(1, opt.max_length, -1)
        return "Set: " + sentence + " -> " + str(words)
    else:
        return "Set failed"


def main():
    api.run(host=args.ip, port=args.port)


if __name__ == '__main__':
    main()
