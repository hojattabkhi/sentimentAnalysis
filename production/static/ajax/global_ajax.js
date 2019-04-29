function getSentenceSentiment() {
    $("#posetive_flag").removeClass("logo-small");
    $("#negative_flag").removeClass("logo-small");
    $("#neural_flag").removeClass("logo-small");
    if ($("#search_filter").val() == '') {
        return;
    }
    $('#wait').show();

    $.ajax({
        url: "/getSentenceSentiment",
        type: "POST",
        data: JSON.stringify({
            'search_filter': $("#search_filter").val(),
        }),
        contentType: "application/json;charset=UTF-8",
        success: function(output, response) {
            result = $.parseJSON(output)
            if(result.status == 'positive'){
                $("#posetive_flag").addClass("logo-small");
            } else if (result.status == 'negative') {
                $("#negative_flag").addClass("logo-small");
            } else {
                $("#neural_flag").addClass("logo-small");
            }
        },
        error: function(error) {
            console.log(error);
        },
        complete: function(){
            $('#wait').hide();
            $('html,body').animate({ scrollTop: $('.container-fluid').position().top }, 'slow');
            document.body.scrollTop = $('.container-fluid').position().top; // For Safari
            document.documentElement.scrollTop = $('.container-fluid').position().top; // For Chrome, Firefox, IE and Opera
        }
    });
}