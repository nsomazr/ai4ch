from django import forms

class ImageHorizontal(forms.Form):
    image_file = forms.FileField(max_length=200, widget=(forms.FileInput(attrs={'class': 'form-control input-file img-responsive img-rounded',
                                                                                'id': 'img_section_one',
                                                                                'name': 'image_file',
                                                                                'placeholder': 'Choose eye image',
                                                                                'onchange': 'form.submit()'})))


