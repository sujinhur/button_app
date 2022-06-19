
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse
from django.core.paginator import Paginator, EmptyPage

import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from pandas import date_range


from .models import StepCount_Data

# Create your views here.

def home(request):
    # 이번주 날짜
    current = datetime.date.today()
    weekday = datetime.datetime.today().weekday()
    startday = current - timedelta(days=weekday)
    # 이번주 DB id
    current_index = StepCount_Data.objects.get(saved_time = current).id
    startday_index = StepCount_Data.objects.get(saved_time = startday).id
    model_index = StepCount_Data.objects.latest('id').id

    # pagination
    page = request.GET.get("page", 1)
    # 이번주 데이터만 보여주기 위함
    if page == 1:
        datalist = StepCount_Data.objects.order_by('-saved_time')[model_index - current_index:model_index - startday_index +1]
        paginator = Paginator(datalist, current_index - startday_index + 1)
    # 저번주 부터 1주일씩 데이터 보여줌  
    else:
        datalist = StepCount_Data.objects.order_by('-saved_time')[model_index - startday_index - 6:]
        paginator = Paginator(datalist, 7)
    # url에 있는 현재 page값 get_page로 전달
    data = paginator.page(page)  
 
    # 날짜 범위 
    month = []
    day = []
    for i in reversed(data.object_list):
        month.append(i.saved_time.month)
        day.append(i.saved_time.day)

        date_range = str(month[0]) + "월 " + str(day[0]) + "일 ~" + str(month[-1]) + "월 " + str(day[-1]) + "일"

    stepcount = []
    for i in reversed(data.object_list):
        stepcount.append(i.stepCount)

    context = {
        "page" : data,
        "date_range" : date_range,
        "stepcount" : stepcount,
    }
    return render(request, 'visstep/home.html', context)


def month(request, num = 1):
    # 이번달 날짜
    current = datetime.date.today()
    current_day = current.day
    startday = current - timedelta(days=current_day -1)

    # next, prev page 번호 및 이에 따른 데이터 불러오기
    next_page_index = num + 1
    if num <= 1:
        prev_page_index = 1
        date = StepCount_Data.objects.filter(saved_time__range=[startday, current])
  
    else:
        prev_page_index = num - 1
        if current.month - num + 1 <= 0:
            year_num = (current.month - num + 1)//(-12)
            month_num = -((current.month - num + 1)%(-12))
            date = StepCount_Data.objects.filter(saved_time__year = str(current.year - year_num - 1), saved_time__month = str(12 - month_num))
            if StepCount_Data.objects.get(saved_time = str(date[0])).id == 1:
                next_page_index = num
        else:
            date = StepCount_Data.objects.filter(saved_time__year = str(current.year), saved_time__month = str(current.month - num + 1))

    days =[]
    stepcount = []
    for i in date:
        days.append(i.saved_time.day)
        stepcount.append(i.stepCount)

    # 날짜 범위
    first_date = StepCount_Data.objects.get(saved_time = str(date[0])).saved_time
    date_range = str(first_date.year) + "년 " + str(first_date.month) + "월" 

    context = {
        "days" : days,
        "stepcount" : stepcount,
        "next_page_index" : next_page_index,
        "prev_page_index" : prev_page_index,
        "date_range" : date_range,
    }
    return render(request, 'visstep/month.html', context)


def year(request, num = 1):
    # 올해 날짜
    current = datetime.date.today()
    current_month = current.month
    current_day = current.day
    startday = current - relativedelta(months=current_month -1) - timedelta(days=current_day -1)

    # next, prev page 번호 및 이에 따른 데이터 불러오기
    next_page_index = num + 1
    if num <= 1:
        prev_page_index = 1
        date = StepCount_Data.objects.filter(saved_time__range=[startday, current])
    else:
        prev_page_index = num - 1
        new_day = startday - relativedelta(years=num -1)
        date = StepCount_Data.objects.filter(saved_time__range=[new_day, new_day + relativedelta(years=1) - timedelta(days=1)])
        if StepCount_Data.objects.get(saved_time = str(date[0])).id == 1:
            next_page_index = num

    month = []
    stepcount = []
    num = 0
    count = 0
    for i in range(1,13):
        for j in date:
            if i == j.saved_time.month:
                num = num + j.stepCount
                count = count + 1
            else:
                if num == 0:
                    break
                stepcount.append(num//count)
                num = 0
                count = 0
                i = i+1


    # 날짜 범위
    first_date = StepCount_Data.objects.get(saved_time = str(date[0])).saved_time
    date_range = str(first_date.year) + "년"


    context = {
        "stepcount" : stepcount,
        "next_page_index" : next_page_index,
        "prev_page_index" : prev_page_index,
        "date_range" : date_range,
    }
    return render(request, 'visstep/year.html', context)

def specify(request):
    current = datetime.date.today()
    weekday = datetime.datetime.today().weekday()
    startday = current - timedelta(days=weekday)

    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    print(start_date)

    if not start_date:
        date = StepCount_Data.objects.filter(saved_time__range = [startday, current])
        date_range = str(startday.month) + "월 " + str(startday.day) + "일 ~ " + str(current.month) + "월 " + str(current.day) + "일"
        
        stepcount = []
        for i in date:
            stepcount.append(i.stepCount)

        context = {
        "date_range" :date_range,
        "stepcount" : stepcount, 

        }
        return render(request, 'visstep/specify.html', context)
    else:
        date = StepCount_Data.objects.filter(saved_time__range = [start_date, end_date])
        print(date)
        first_date = StepCount_Data.objects.get(saved_time = str(date[0])).saved_time
        final_date = StepCount_Data.objects.get(saved_time = str(date[len(date)-1])).saved_time
        date_range = str(first_date.month) + "월 " + str(first_date.day) + "일 ~ " + str(final_date.month) + "월 " + str(final_date.day) + "일"
        context = {
        "date_range" :date_range,
        "date" : list(date.values()), 

        }   
        return JsonResponse(context)
    


def compare(request):
    current = datetime.date.today()
    weekday = datetime.datetime.today().weekday()
    startday = current - timedelta(days=weekday)

    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    print(start_date)

    if not start_date:

        date_1 = StepCount_Data.objects.filter(saved_time__range = [startday, current])
        date_2 = StepCount_Data.objects.filter(saved_time__range = [startday - timedelta(days= 7), startday - timedelta(days= 1)])


        date_range_1 = str(startday.month) + "월 " + str(startday.day) + "일 ~ " + str(current.month) + "월 " + str(current.day) + "일"
        date_range_2 = str(date_2[1].saved_time.month) + "월 " + str(date_2[1].saved_time.day) + "일 ~ " + str(date_2[6].saved_time.month) + "월 " + str(date_2[6].saved_time.day) + "일"

        context = {
            "date_range_1" :date_range_1,
            "date_range_2" :date_range_2,
            "date_1" : date_1,
            "date_2" : date_2,
        }
        return render(request, 'visstep/compare.html', context)
    
    else:
        date = StepCount_Data.objects.filter(saved_time__range = [start_date, end_date])
        print(date)
        first_date = StepCount_Data.objects.get(saved_time = str(date[0])).saved_time
        final_date = StepCount_Data.objects.get(saved_time = str(date[len(date)-1])).saved_time
        date_range = str(first_date.month) + "월 " + str(first_date.day) + "일 ~ " + str(final_date.month) + "월 " + str(final_date.day) + "일"
        context = {
        "date_range" :date_range,
        "date" : list(date.values()), 

        }   
        return JsonResponse(context)