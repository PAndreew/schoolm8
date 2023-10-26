from tortoise import fields
from tortoise.models import Model

class TopicDataMapping(Model):
    id = fields.IntField(pk=True)
    book_id = fields.CharField(max_length=255)
    book_title = fields.CharField(max_length=255)
    topic = fields.JSONField()

