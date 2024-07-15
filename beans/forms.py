from django import forms
from .models import BeansData

class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={
            'class': 'form-control input-file img-responsive img-rounded',
            'id': 'img_section_one',
            'name': 'image_file',
        }))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result

class UploadForm(forms.ModelForm):
    file = MultipleFileField(label='Select files', required=False,)

    class Meta:
        model = BeansData
        fields = ['file', ]