var parseGenerateReplyJson = function (data) {
    var json = {};

    for (idx = 0; idx < data.length; idx++) {
        json[data[idx].name] = data[idx].value;
    }
    return json;
};
var sendGenerate = function () {
    var data = parseGenerateReplyJson($('form.generate').serializeArray());

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
                $('#display').text(data.reply);
            })

            .fail(function (xhr, textStatus, errorThrown) {
                var text = JSON.stringify({
                        "XMLHttpRequest": xhr,
                        "textStatus": textStatus,
                        "errorThrown": errorThrown
                    },
                    undefined, 2
                );
                $('#display').text(text);
            })
    } catch (e) {
        alert(e);
        return false;
    }

};
