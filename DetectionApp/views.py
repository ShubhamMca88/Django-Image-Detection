from django.shortcuts import render
from .forms import ImageUploadForm
from .detection_model import detect_objects
import os

def image_upload_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = os.path.join('media', image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            result = detect_objects(image_path)
            os.remove(image_path)

            return render(request, 'result.html', {'result': result})
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})
