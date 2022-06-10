from django.shortcuts import render
from django.http import HttpResponse

import datetime
from datetime import timedelta

from .models import StepCount_Data

# Create your views here.

def home(request):
    current = datetime.date.today()
    weekday = datetime.datetime.today().weekday()
    print(weekday)
    startday = current - timedelta(days=weekday)
    print(startday)

    dataset = StepCount_Data.objects.raw("SELECT * FROM stepcountData WHERE saved_time BETWEEN '" + str(startday) + "' and '" + str(current) +"'")
    # dataset = StepCount_Data.objects.raw("SELECT * FROM stepcountData WHERE saved_time BETWEEN '" + str(startday- timedelta(days=7)) + "' and '" + str(startday- timedelta(days=1)) +"'")

    date = []
    stepcount = []
    for i in dataset:
        date.append(i)
        stepcount.append(i.stepCount)
    print(type(date[0]))

    day = str(startday.month) + "월 " + str(startday.day) + "일 ~ " + str(current.month) + "월 " + str(current.day) + "일"
     
    context = {
        'date': date,
        'stepcount' : stepcount,
        'day': day
    }

    return render(request, 'visstep/home.html', context)

def month(request):
    return render(request, 'visstep/month.html')

def year(request):
    return render(request, 'visstep/year.html')