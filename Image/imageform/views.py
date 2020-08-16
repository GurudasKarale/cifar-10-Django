from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def home(request):
    return render(request,"index.html")

def getimage(request):

    a={0:"Airplane",1:"Automobile",2:"Bird",3:"Cat",4:"Deer",5:"Dog",6:"Frog",7:"Horse",8:"Ship",9:"Truck"}

    if request.method=='POST':
        f=request.FILES['image']
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)

        img = load_img(file_url, target_size=(32, 32))
        img = img_to_array(img)
        img = img.reshape(1, 32, 32, 3)
        img = img.astype('float32')
        img = img / 255.0

        model = load_model('cifmodel.h5')
        result = model.predict_classes(img)
        return render(request,"index.html",{'result':'Voila! The uploaded image is of:','response':a[result[0]]})







