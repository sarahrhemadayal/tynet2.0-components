from django.urls import path
from .views import TrashDataView

urlpatterns = [
    path("trash-data/", TrashDataView.as_view(), name="trash-data"),
]
