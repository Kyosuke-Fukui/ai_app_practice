from django.shortcuts import render
# from django.http import HttpResponse
import pandas as pd
import pickle

category_data = pd.read_csv("idx2category.csv")
idx2category = {row.k: row.v for idx, row in category_data.iterrows()}

with open("rdmf.pickle", mode="rb") as f:
    model = pickle.load(f)

# Create your views here.

# def index(request):
#     html = """<h1>Hello World</h1>
#     <input type="text">
#     """
#     return HttpResponse(html)

def index(request):
    if request.method == "GET":
        return render(request,"nlp/home.html")

    #POSTされた場合
    else:
        title = [request.POST["title"]]
        result = model.predict(title)[0] #カテゴリを表すインデックス値が返される
        pred = idx2category[result]
        return render(
            request,
            "nlp/home.html",
            {"title":pred }
        )

    