#!/bin/bash

PORT=5050
IP=0.0.0.0

function usage {
    cat <<EOM
Usage: $(basename "$0") [OPTION]...
    -p              ポート番号を指定
    -i              IPまたはHostを指定
    -h, --help      このヘルプを表示
EOM
    exit 2
}

while getopts ":p:h" optKey; do
    case "$optKey" in
        p)
            PORT=${OPTARG}
            ;;
        i)
            IP=${OPTARG}
            ;;
        '-h'|'--help'|* )
        usage
        ;;
    esac
done

cd app

if [ ! -e ./data/style_sensitive_dict.bin ]; then
    cat ./data/style_sensitive_dict.bin00 ./data/style_sensitive_dict.bin01 > ./data/style_sensitive_dict.bin
    echo "Concat style-sensitive dict"
fi

pipenv run python -m persona_chatbot_app.bot.server -p ${PORT} -i ${IP}
