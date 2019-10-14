
function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                $('#gesture')
                    .attr('src', e.target.result)
                    .width(300)
                    .height(300);
            };

            reader.readAsDataURL(input.files[0]);
        }
    }
function validate(){
  if(document.getElementById("file").files.length == 0){
    alert("Please select an image for prediction!");
    return;
  }
}
