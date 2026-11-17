from django.http import HttpResponse
from django.shortcuts import render


def feedback(request):
    # A text input named "message" is submitted from the form below.
    message = request.GET.get('message', '')
    # The submitted text is concatenated straight into the HTML response.
    html = '<div class="note"><h2>Your feedback</h2><p>' + message + '</p></div>'
    return HttpResponse(html)


def feedback_form(request):
    # Renders the form that has the text input.
    return render(request, 'feedback.html')
