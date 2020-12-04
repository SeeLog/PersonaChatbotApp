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

## フロントエンド実装の例について
`app/persona_chatbot_app/bot/templates/index.html` や `app/persona_chatbot_app/bot/static/` 以下がそれに当たります．

Flaskによって自動起動するように設定してあるので，上記の方法でサーバを実行した上でlocalhost等にアクセスするとフロントエンド実装が見れると思います．

つまり
```
% ./start_server_python.sh -p 8080
```
などとしてから
`http://localhost:8080`にアクセスすれば良いというわけです．
ポート開放等の設定が必要になるかもしれません．

とくにGPUマシン上でサーバを立ち上げた場合，ポートフォワーディングなどをするとよいでしょう．

## その他オプションについて
```
% cd app
% pipenv run python -m persona_chatbot_app.bot.server --help
```
で見れます．

## モデルについて
このリポジトリではモデルの学習等については管理していません．
学習済みモデルを持ってきて使用しています．
