var parseToJson = function (data) {
    var json = {};

    for (idx = 0; idx < data.length; idx++) {
        json[data[idx].name] = data[idx].value;
    }
    return json;
};

var showAlert = function(text) {
    alert_obj = $('.alert-top' + '.alert-danger');
    alert_obj.text(text);
    alert_obj.fadeIn(500);
    alert_obj.delay(2000).fadeOut(1000);
}

var showSuccess = function(text) {
    alert_obj = $('.alert-top' + '.alert-success');
    alert_obj.text(text);
    alert_obj.fadeIn(500);
    alert_obj.delay(2000).fadeOut(1000);
}

var showWords = function() {
    words_obj = $('#extract-result');
    toggle = $(".slide-toggle")
    words_obj.fadeIn(500);
    toggle.fadeIn(500);
}

var hideWords = function() {
    words_obj = $('#extract-result');
    toggle = $(".slide-toggle")
    words_obj.fadeOut(500);
    //toggle.fadeOut(500);
}

let valueMax = 50;
let valueMin = -50;

let vectorChanged = false;

var vecInputList = [];
var personaWords = [];
var curVector = [];

var getSlideBar = function(data, i) {
    var sliderCls = "vec-slider-" + i.toString();
    var inputName = "vec" + i.toString();

    return {
        'tag': '<div class="form-inline"><label for="vec' + i.toString() + '">' + i.toString() +
                ': </label><input type="number" style="width: 40%;" class="form-control" name="vec' + i.toString() +
                '" min="' + valueMin.toString() + '" max="' + valueMax.toString() + '" value="' + data["persona_vector"][i] + '" idx="' + i.toString() + '">' +
                '<input type="range" style="width: 40%;" min="' + valueMin.toString() +
                '" max="' + valueMax.toString() + '" step="0.0000000000000001" class="' + sliderCls +
                '" target="vec'+ i.toString() + '" value="' + data["persona_vector"][i] + '" idx="' + i.toString() + '">' +
                '</div>',
        'cls': sliderCls,
        'name': inputName
    }
}

var appendSlideBars = function(data) {
    if (data != null && data["persona_dim"] != null) {
        let count = data["persona_dim"];

        var area = $('.slide-bar');

        area.text("");
        vecInputList = [];

        for (let i = 0; i < count; i++) {
            let ipt = getSlideBar(data, i);
            area.append(ipt["tag"]);

            vecInputList.push(ipt["cls"]);

            $('.' + ipt["cls"]).on('input',
                function(event) {
                    // スライドバーを操作時
                    var val = parseFloat(event.target.value);
                    if (val != NaN) {
                        $('input[name=' + $(event.target).attr('target') + ']').val(val);
                        let idx = parseInt($(event.target).attr('idx'));
                        curVector[idx] = val;
                        vectorChanged = true;
                    }
                });
            $('input[name=' + ipt['name']).on('input',
                function(event) {
                    // テキストを操作時
                    var val = parseFloat(event.target.value);
                    if (val != NaN) {
                        $('input[target=' + $(event.target).attr('name') + ']').val(val);
                        let idx = parseInt($(event.target).attr('idx'));
                        curVector[idx] = val;
                        vectorChanged = true;
                    }
                });
        }
    }
    else {
        showAlert("抽出に失敗しているみたいです");
    }
}

var sendGenerate = function () {
    var data = parseToJson($('form.generate').serializeArray());

    if (data["sentence"] == "") {
        showAlert("無言はダメです！");
        return false;
    }

    var right_box = makeRightBox(data["sentence"]);
    $(right_box).appendTo('.chat-frame').hide().fadeIn(500);

    try {
        $.ajax({
                url: $('form.generate').attr("action"),
                type: 'POST',
                dataType: 'json',
                contentType: 'application/json',
                scriptCharset: 'utf-8',
                data: JSON.stringify(data)
            })

            .done(function (data, textStatus, xhr) {
                $('#send-sentence').val("");
                var left_box = makeLeftBox(data.reply);
                $(left_box).appendTo('.chat-frame').hide().fadeIn(750);
                var main_obj = $('main');
                main_obj.scrollTop(main_obj.prop("scrollHeight"));
            })

            .fail(function (xhr, textStatus, errorThrown) {
                var text = JSON.stringify({
                        "XMLHttpRequest": xhr,
                        "textStatus": textStatus,
                        "errorThrown": errorThrown
                    },
                    undefined, 2
                );
                //$('.chat-frame').text(text);
                showAlert("生成に失敗しました．サーバは起動していますか？")
            })
    }
    catch (e) {
        showAlert(e);
        return false;
    }
};

var makeRightBox = function(sentence) {
    return '<div class="speech-bubble"><div class="sb-bubble sb-flat sb-right">' + sentence + '</div></div>';
}

var makeLeftBox = function(reply) {
    return '<div class="faceicon faceleft"><img src="/static/img/robot.png" alt=""></div><div class="speech-bubble"><div class="sb-bubble sb-flat sb-left">'
        + reply + '</div></div>';
}


var getPersonaFromSentence = function() {
    var data = parseToJson($('form.get-persona').serializeArray());

    if (data["sentence"] == "") {
        showAlert("無言はダメです！");
        return false;
    }

    try {
        $.ajax({
                url: $('form.get-persona').attr("action"),
                type: "POST",
                dataType: "json",
                contentType: "application/json",
                scriptCharset: "utf-8",
                data: JSON.stringify(data)
            }
        )

        .done(function (data, textStatus, xhr) {
            if (data["success"] == true) {
                $("#extracted").text(data["words"]);
                personaWords = data["words"];
                vectorChanged = false;
                curVector = data["persona_vector"];
                showWords();
                appendSlideBars(data);
            }
            else {
                hideWords();
                showAlert("ペルソナに用いる単語の抽出に失敗しました．未知語などのチェックをしてみてください．");
            }
        })

        .fail(function (xhr, textStatus, errorThrown) {
            showAlert("ペルソナに用いる単語の抽出に失敗しました．サーバは起動していますか？")
        })
    }
    catch (e) {
        showAlert(e);
        return false;
    }
}

var setPersona = function() {
    var json = {
        "should_return_vector": true
    }

    var sentence = $('#persona-sentence').val();

    if (vectorChanged) {
        json["persona_vector"] = curVector;
    }
    else {
        if (sentence == undefined || sentence == null || sentence == "") {
            showAlert("無言はダメです！" + sentence);
            return false;
        }
        json["sentence"] = sentence;
    }

    try {
        $.ajax({
            url: $('form.set-persona').attr("action"),
            type: "POST",
            dataType: "json",
            contentType: "application/json",
            scriptCharset: "utf-8",
            data: JSON.stringify(json)
            }
        )

        .done(function (data, textStatus, xhr) {
            if (data["success"] == true) {
                $("#extracted").text(data["words"]);
                personaWords = data["words"];
                if (personaWords == null) {
                    vectorChanged = true;
                    hideWords();
                }
                else {
                    vectorChanged = false;
                    showWords();
                }
                curVector = data["persona_vector"];
                appendSlideBars(data);
                // 成功
                showSuccess("ペルソナのセットに成功しました！");
                $('#config-modal').modal('hide');
            }
            else {
                hideWords();
                showAlert("ペルソナのセットに失敗しました．");
            }
        })

        .fail(function (xhr, textStatus, errorThrown) {
            showAlert("ペルソナのセットに失敗しました．サーバは起動していますか？")
        })
    }
    catch (e) {
        showAlert(e);
    }
}

var getCurrentPersona = function() {
    try {
        $.ajax({
            url: $('form.get-current-persona').attr("action"),
            type: "GET",
            contentType: "application/json",
            scriptCharset: "utf-8"
        })

        .done(function (data, textStatus, xhr) {
            if (data["persona_dim"] != null) {
                personaWords = data["words"];

                curVector = data["persona_vector"];

                if (personaWords != null && personaWords.length > 0) {
                    $("#extracted").text(data["words"]);
                    showWords();
                }
                else {
                    $("#extracted").text("");
                    hideWords();
                }

                appendSlideBars(data);
            }
        })
        .fail(function (xhr, textStatus, errorThrown) {
            showAlert("サーバの状態の取得に失敗しました．サーバは起動していますか？")
        })
    }
    catch (e) {
        showAlert(e);
    }
}
