from django.db import models

# Create your models here.

class StepCount_Data(models.Model):
    saved_time = models.DateField()
    stepCount = models.IntegerField()

    def __str__(self):
        return str(self.saved_time)

    class Meta:
        db_table = 'stepcountData'