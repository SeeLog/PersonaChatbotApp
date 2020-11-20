#!/bin/bash


cd app

pipenv run pyinstaller persona_chatbot_app/bot/server.py --onefile

cp ./dist/server ./
