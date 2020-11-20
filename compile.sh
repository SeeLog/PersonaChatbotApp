#!/bin/bash


cd app

pipenv run pyinstaller ./server.spec

cp ./dist/server ./
