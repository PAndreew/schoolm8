from typing import List, Dict
from database.models import TopicDataMapping

from typing import List, Dict
from database.models import TopicDataMapping
import logging

logger = logging.getLogger(__name__)

async def save_topics(book_id: str, book_title: str, lda_result: List[Dict]):
    try:
        for topic in lda_result:
            topic_data = TopicDataMapping(
                book_id=book_id,
                book_title=book_title,  # Include the book title
                topic=topic,
            )
            await topic_data.save()
        logger.info(f"Successfully saved topics for book_id: {book_id}")
    except Exception as e:
        logger.error(f"Failed to save topics for book_id: {book_id}. Error: {e}")
        raise


