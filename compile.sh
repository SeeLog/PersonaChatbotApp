#!/bin/bash


cd app

pipenv run python -m PyInstaller ./server.spec

cp ./dist/chatbot_server ./
