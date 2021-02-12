# Persona Chatbot App

## はじめに
### これは何？
- 現実的に調整可能な数の発話内容を制御できるパラメータを持つChatbotソフトウェアです
- 基本的にサーバサイドで動作させることを念頭に置いています
- APIで呼ぶことができます
- フロントエンド実装の例もおまけとしてつけてあります

## 導入方法
### pipenvを導入する
各自で最低限Python3.8以上の環境を用意してください
```
% pip install pipenv
```
### 環境を構築する
以下のコマンドで必要なPythonライブラリが入ったpipenv環境が構築されます
```
% pipenv install
```
また，MeCabとMeCab用の辞書でIPADicのUTF-8のものが入っていない場合，正常に起動ができません．
その際は `install_mecab_example.sh` を参考にMeCabと辞書をインストールしてください．
Ubuntu16.04の場合はそのまま使えるかもしれません．
```
./install_mecab_example.sh
```
### サーバを実行する
```
% ./start_server_python.sh
```
or
```
% cd app
% pipenv run python -m persona_chatbot_app.bot.server -p [ポート番号] -i [IP or Host]
```
`-p` や `-i` を省略した場合は `-p 5050`, `-i 0.0.0.0` がデフォルトで用いられます．

## APIについて
APIはopenapiを用いて記述してあります．
`openapi.yaml`を参考にしてください．

または `redoc-cli` を用いて静的HTML化したものを `APIReference.html` として用意してありますので，そちらをご覧ください．

## フロントエンド実装の例について
`app/persona_chatbot_app/bot/templates/index.html` や `app/persona_chatbot_app/bot/static/` 以下がそれに当たります．

Flaskによって自動起動するように設定してあるので，上記の方法でサーバを実行した上でChromeでlocalhost等にアクセスするとフロントエンド実装が見れると思います．
Chromeを想定した実装になっているので他のブラウザだと表示がブレるかもしれません．


## その他オプションについて
```
% cd app
% pipenv run python -m persona_chatbot_app.bot.server --help
```
で見れます．

## モデルについて
最大ファイル制限内に収まるファイルサイズでしたので学習済みモデルもcloneするだけで用いることができます．
さらにペルソナの最小の次元数を変更したい場合や，その他のモデルを用いたい場合は[こちら](https://drive.google.com/drive/folders/1oamKariMb2hx-RBrY-rfUqomFpwbu36O?usp=sharing)をご利用ください．
