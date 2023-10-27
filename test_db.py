import asyncio
from tortoise import Tortoise, fields
from tortoise.models import Model

class TopicDataMapping(Model):
    id = fields.IntField(pk=True)
    book_id = fields.CharField(max_length=255)
    book_title = fields.CharField(max_length=255)
    topic = fields.JSONField()

async def run():
    # Initialize Tortoise ORM
    await Tortoise.init(
        db_url='sqlite://topics.db',
        modules={'models': ['__main__']}
    )
    await Tortoise.generate_schemas()

    # Execute the query
    query = await TopicDataMapping.all().values('book_title')
    print(query)

    # Close the ORM connection
    await Tortoise.close_connections()

# Run the script
loop = asyncio.get_event_loop()
loop.run_until_complete(run())
