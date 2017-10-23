var cont=document.getElementsByTagName("body")[0];
var imgs=document.getElementsByTagName("a");
var i=0;
//var divv= document.createElement("div");
var input = document.createElement("textarea");
input.name="mytxtarea";
input.cols="100";
input.rows="50";
//input.maxLength="60000";
var aray=new Array();var j=-1;
while(++i<imgs.length){
    if(imgs[i].href.indexOf("/imgres?imgurl=http")>0){
      //divv.appendChild(document.createElement("br"));
      aray[++j]=decodeURIComponent(imgs[i].href).split(/=|%|&/)[1].split("?imgref")[0];
      //divv.appendChild(document.createTextNode(aray[j]));
      input.value+=String(aray[j])+"\n";
    }
 }
//cont.insertBefore(divv,cont.childNodes[0]);
cont.insertBefore(input,cont.childNodes[0]);
document.getElementsByClassName('sfbg nojsv')[0].style.display = 'none';
document.getElementsByClassName('tsf-p')[0].style.display='none';
