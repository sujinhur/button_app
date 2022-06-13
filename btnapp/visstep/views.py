from multiprocessing.reduction import steal_handle
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.core.paginator import Paginator, EmptyPage

import datetime
from datetime import timedelta


from .models import StepCount_Data

# Create your views here.

def home(request):
    current = datetime.date.today()
    weekday = datetime.datetime.today().weekday()
    startday = current - timedelta(days=weekday)

    current_index = StepCount_Data.objects.get(saved_time = current).id
    startday_index = StepCount_Data.objects.get(saved_time = startday).id
    model_index = StepCount_Data.objects.latest('id').id


    page = request.GET.get("page", 1)

    if page == 1:
        datalist = StepCount_Data.objects.order_by('-saved_time')[model_index - current_index:model_index - startday_index +1]
        paginator = Paginator(datalist, current_index - startday_index + 1)
        
    else:
        datalist = StepCount_Data.objects.order_by('-saved_time')[model_index - startday_index - 6:]
        paginator = Paginator(datalist, 7)

    data = paginator.page(page)  
 
    month = []
    day = []
    for i in reversed(data.object_list):
        month.append(i.saved_time.month)
        day.append(i.saved_time.day)

    date = str(month[0]) + "월 " + str(day[0]) + "일 ~" + str(month[-1]) + "월 " + str(day[-1]) + "일"

    context = {
        "page" : data,
        "page_list" : reversed(data.object_list),
        "date" : date,
    }
    return render(request, 'visstep/home.html', context)

# def home(request):
#     current = datetime.date.today()
#     weekday = datetime.datetime.today().weekday()
#     print(weekday)
#     startday = current - timedelta(days=weekday)
#     print(startday)

#     dataset = StepCount_Data.objects.raw("SELECT * FROM stepcountData WHERE saved_time BETWEEN '" + str(startday) + "' and '" + str(current) +"'")
#     # dataset = StepCount_Data.objects.raw("SELECT * FROM stepcountData WHERE saved_time BETWEEN '" + str(startday- timedelta(days=7)) + "' and '" + str(startday- timedelta(days=1)) +"'")

#     date = []
#     stepcount = []
#     for i in dataset:
#         date.append(i)
#         stepcount.append(i.stepCount)
#     print(type(date[0]))

#     day = str(startday.month) + "월 " + str(startday.day) + "일 ~ " + str(current.month) + "월 " + str(current.day) + "일"
     
#     context = {
#         'date': date,
#         'stepcount' : stepcount,
#         'day': day
#     }

#     return render(request, 'visstep/home.html', context)

def month(request):
    current = datetime.date.today()
    current_day = datetime.datetime.today().day
    startday = current - timedelta(days=current_day -1)
    print(startday)

    current_index = StepCount_Data.objects.get(saved_time = current).id
    startday_index = StepCount_Data.objects.get(saved_time = startday).id
    model_index = StepCount_Data.objects.latest('id').id

    context = {

    }
    return render(request, 'visstep/month.html', context)

def year(request):
    return render(request, 'visstep/year.html')