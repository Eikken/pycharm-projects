
    $(function () {
        getcity();
    });

    function getcity() {
        //        var province = returnCitySN.province;
//        var city = returnCitySN.cname;
//        $("#hfcity").val(city);
    }

    function loginform() {
        var info = new Object();
        info.UserId = $("#UserId").val();
        info.Password = $("#Password").val();
        info.VeriCode = $("#VeriCode").val();
        info.city = $("#hfcity").val();
        var jsonObject = JSON.stringify(info);
//        var key = CryptoJS.enc.Utf8.parse("8NONwyJtHesysWpM");
//        var encrypjsonObject = CryptoJS.AES.encrypt(jsonObject, key, {
//            mode: CryptoJS.mode.ECB,
//            padding: CryptoJS.pad.Pkcs7
//        });
//        var jsonstr = encrypjsonObject.ciphertext.toString()
        var rsa = new JSEncrypt();
        var pubkey = $("#pubkey").val();
        rsa.setPublicKey(pubkey);
        var jsonstr = rsa.encrypt(jsonObject);
        $.ajax({
            type: 'post',
            url: "stulogin_do",
            dataType: "json", //返回json格式的数据
            data: { 'json': jsonstr },
            cache: false,
            success: function (data) {
                if (data.jg == "1") {
                    document.location = "../" + data.url;
                }
                else {
                    $('#imgVerifi').attr('src', '/Home/VerificationCode' + '?codetype=stucode&&t=' + new Date().getSeconds());
                    alert(data.msg);
                    //                    $.messager.alert("提示", data.msg, "error");
                }
            },
            error: function (e) {
                alert(e.toString());
                //                $.messager.alert("提示", e.toString(), "error");
            }
        });
    }
    window.onkeypress = function () {
        if (event.keyCode == 13) {
            document.getElementById('btnlogin').click(); //btnOK为登录按钮的ID
        }
    }
    function changecode() {
        $('#imgVerifi').attr('src', '/Home/VerificationCode' + '?codetype=stucode&t=' + new Date().getTime());
    }
