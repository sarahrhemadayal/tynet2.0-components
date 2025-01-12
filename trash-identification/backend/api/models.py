from django.db import models

class TrashData(models.Model):
    month = models.CharField(max_length=20)
    biodegradable_percentage = models.FloatField(default=0.0)
    recyclable_percentage = models.FloatField(default=0.0)
    hazardous_percentage = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.month}: Biodegradable={self.biodegradable_percentage}%, Recyclable={self.recyclable_percentage}%, Hazardous={self.hazardous_percentage}%"
