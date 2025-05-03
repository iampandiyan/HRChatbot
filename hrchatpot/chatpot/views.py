from django.shortcuts import render, redirect
from chatpot.utils.query_engine import answer_query
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import ZipUploadForm
from .utils.embedding_loader import process_policy_documents 
import os, zipfile
from django.contrib.auth.decorators import login_required  
from django.contrib.auth.decorators import user_passes_test
import shutil

@login_required
def index(request):
    # Clear session when visiting the homepage directly
    request.session["chat_history"] = []
    return render(request, 'chatpot/index.html', {'chat_history': []})

@login_required
def ask_question(request):
    if request.method == "POST":
        if "chat_history" not in request.session:
            request.session["chat_history"] = []

        question = request.POST.get("question")
        answer = answer_query(question)  # Your logic to get the answer

        # Update session history
        request.session["chat_history"] += [
            {"role": "user", "message": question},
            {"role": "bot", "message": answer}
        ]
        request.session.modified = True

        # Send back only the bot's message as a div
        return HttpResponse(f'<div class="message bot">{answer}</div>')

    return HttpResponse(status=405)

def login_view(request):
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = CustomLoginForm()
    return render(request, 'login.html', {'form': form})

@login_required
def dashboard(request):
    print(f"User role: {request.user.role}")
    return render(request, 'chatpot/dashboard.html', {'role': request.user.role})

def dashboardTest(request):
    return render(request, 'chatpot/dashboard.html')

def is_hr(user):
    return user.role == 'HR'


@login_required
@user_passes_test(is_hr)
def upload_zip(request):
    message = ""
    success = False

    if request.method == 'POST':
        form = ZipUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                zip_file = form.cleaned_data['zip_file']
                extract_to = os.path.join('chatpot', 'uploaded_docs')

                # Create or clean the folder
                os.makedirs(extract_to, exist_ok=True)
                os.chmod(extract_to, 0o777)
                for filename in os.listdir(extract_to):
                    file_path = os.path.join(extract_to, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

                # Extract the zip
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)

                process_policy_documents(extract_to)
                message = "Upload successful!"
                success = True
            except Exception as e:
                print(f"Upload failed: {e}")
                message = f"Upload failed: {e}"
        else:
            message = "Invalid form submission."
    else:
        form = ZipUploadForm()

    return render(request, 'chatpot/upload.html', {'form': form, 'message': message, 'success': success})

