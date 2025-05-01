from django.shortcuts import render
from chatpot.utils.query_engine import answer_query
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


def index(request):
    # Clear session when visiting the homepage directly
    request.session["chat_history"] = []
    return render(request, 'chatpot/index.html', {'chat_history': []})

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
