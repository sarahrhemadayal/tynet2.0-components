from rest_framework.views import APIView
from rest_framework.response import Response
from .models import TrashData

class TrashDataView(APIView):
    def get(self, request):
        data = TrashData.objects.all().values()
        return Response(data)

    def post(self, request):
        data = request.data
        trash_data, created = TrashData.objects.update_or_create(
            month=data["month"],
            defaults={
                "biodegradable_percentage": data["biodegradable_percentage"],
                "recyclable_percentage": data["recyclable_percentage"],
                "hazardous_percentage": data["hazardous_percentage"],
            }
        )
        return Response({"status": "success", "data": trash_data.id})
