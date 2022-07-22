from django.db import models

# Create your models here.

class StepCount_Data(models.Model):
    date = models.DateField()
    stepCount = models.IntegerField()

    def __str__(self):
        return str(self.date)

    class Meta:
        db_table = 'stepcountData'