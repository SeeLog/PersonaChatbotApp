#!/bin/bash

PORT=5000

function usage {
    cat <<EOM
Usage: $(basename "$0") [OPTION]...
    -p, --port      ポート番号を指定
    -h, --help      このヘルプを表示
EOM
    exit 2
}

while getopts ":p:h" optKey; do
    case "$optKey" in
        p|'--port')
            PORT=${OPTARG}
            ;;
        '-h'|'--help'|* )
        usage
        ;;
    esac
done

cd app
python -m persona_chatbot_app.bot.server -p ${PORT}
