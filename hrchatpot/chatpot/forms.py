from django import forms
from django.contrib.auth.forms import AuthenticationForm

class ZipUploadForm(forms.Form):
    zip_file = forms.FileField(label='Select a ZIP file')

class CustomLoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))
