from database.models import Topic

async def save_topics(book_id: str, lda_result):
    topic = Topic(book_id=book_id, lda_result=lda_result)
    await topic.save()
