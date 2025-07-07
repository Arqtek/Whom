from typing import List, Dict, Optional
import aiohttp
from pydantic.v1 import BaseModel
from uagents import Agent, Context, Model
from groq import Groq
import logging
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import uuid
from dotenv import load_dotenv

load_dotenv()

import os

agent = Agent(name="manager", seed="secret_seed_phrase_concept",
              endpoint="127.0.0.1", port=8000)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Replace with your API Key
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize clients
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize local embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight local model
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None


async def analyze_business_with_llm(business_description: str) -> str:
    """Enhanced business analysis prompt"""
    prompt = f"""
    Analyze the following business description and provide strategic insights for customer research:

    Business Description: "{business_description}"

    Provide a comprehensive JSON response with:
    1. "business_type": Specific category/industry of this business
    2. "target_audience": Detailed customer demographics and psychographics
    3. "key_pain_points": Top 5 problems customers face in this industry
    4. "search_focus": Specific themes to look for in customer discussions
    5. "competitor_keywords": Common terms customers use when comparing options
    6. "emotional_triggers": What emotions drive customer decisions in this space

    Focus on actionable insights that will help identify valuable customer feedback patterns.
    Return only valid JSON without any additional text.
    """

    chat_completion = groq.chat.completions.create(
        messages=[
            {"role": "system",
             "content": "You are a senior business analyst specializing in customer research and market intelligence. Provide detailed, actionable insights in valid JSON format."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
        temperature=0.3,
        max_tokens=500
    )

    analysis = chat_completion.choices[0].message.content
    logger.info(f"Business analysis completed: {analysis}")
    return analysis


def generate_search_queries_with_llm(business_description: str) -> List[str]:
    """Enhanced search query generation with better prompts"""
    try:
        prompt = f"""
        Business Context: "{business_description}"

        Generate 5 highly specific YouTube search queries designed to find videos with rich customer discussions and authentic feedback.

        Target these content types:
        - Customer review videos with active comment sections
        - Problem-solving tutorials where users share experiences
        - Comparison videos that generate debate
        - "Day in the life" or use-case videos
        - Complaint/rant videos about industry issues

        Requirements:
        - Use emotional keywords (frustrated, disappointed, amazing, game-changer)
        - Include comparison terms (vs, better than, alternative to)
        - Add time-based modifiers (2024, recent, new)
        - Focus on customer voice keywords (review, experience, opinion)
        - Avoid overly broad terms

        Return exactly 5 search queries, one per line, no numbering or extra text.
        Don introduce the queries you return with a header like "Here are ..."!!!
        """

        chat_completion = groq.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "You are a YouTube search optimization expert. Create queries that maximize finding authentic customer discussions and detailed feedback in video comments."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=300
        )

        queries = chat_completion.choices[0].message.content.strip().split('\n')
        queries = [q.strip() for q in queries if q.strip()]

        logger.info(f"Generated {len(queries)} search queries: {queries}")
        return queries[:3]

    except Exception as e:
        logger.error(f"Error generating search queries with LLM: {str(e)}")
        return []


class VideoInfo(BaseModel):
    video_id: str
    title: str
    url: str
    channel: str
    description: Optional[str] = None
    duration: Optional[str] = None
    view_count: Optional[str] = None
    upload_date: Optional[str] = None


class CommentInfo(BaseModel):
    comment_id: str
    text: str
    author: str
    like_count: int
    published_at: str
    summary: Optional[str] = None
    summary_embedding: Optional[List[float]] = None
    reply_count: int


def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate vector embedding for text using local model"""
    if embedding_model is None:
        logger.warning("Embedding model not available")
        return None

    try:
        # Generate embedding
        embedding = embedding_model.encode(text)
        # Convert to list for JSON serialization
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


async def search_youtube_videos(query: str, max_results: int = 5) -> List[VideoInfo]:
    """Search for YouTube videos using YouTube Data API v3"""
    try:
        params = {
            'key': YOUTUBE_API_KEY,
            'q': query,
            'part': 'snippet',
            'type': 'video',
            'maxResults': max_results,
            'order': 'relevance'
        }

        url = f"{YOUTUBE_API_BASE_URL}/search"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"YouTube API search failed with status {response.status}")
                    return []

                data = await response.json()

                videos = []
                for item in data.get('items', []):
                    video_id = item['id']['videoId']
                    snippet = item['snippet']

                    # Get additional video details
                    video_details = await get_video_details(video_id)

                    video_info = VideoInfo(
                        video_id=video_id,
                        title=snippet['title'],
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        channel=snippet['channelTitle'],
                        description=snippet.get('description', ''),
                        duration=video_details.get('duration'),
                        view_count=video_details.get('viewCount'),
                        upload_date=snippet['publishedAt']
                    )
                    videos.append(video_info)

                logger.info(f"Found {len(videos)} videos for query: {query}")
                return videos

    except Exception as e:
        logger.error(f"Error searching YouTube: {str(e)}")
        return []


async def get_video_details(video_id: str) -> Dict:
    """Get detailed video information"""
    try:
        params = {
            'key': YOUTUBE_API_KEY,
            'id': video_id,
            'part': 'statistics,contentDetails'
        }

        url = f"{YOUTUBE_API_BASE_URL}/videos"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return {}

                data = await response.json()

                if not data.get('items'):
                    return {}

                item = data['items'][0]
                statistics = item.get('statistics', {})
                content_details = item.get('contentDetails', {})

                return {
                    'viewCount': statistics.get('viewCount', '0'),
                    'likeCount': statistics.get('likeCount', '0'),
                    'commentCount': statistics.get('commentCount', '0'),
                    'duration': content_details.get('duration', '')
                }

    except Exception as e:
        logger.error(f"Error getting video details: {str(e)}")
        return {}


async def scrape_video_comments(video_info: VideoInfo, max_comments: int = 100) -> List[CommentInfo]:
    """Scrape video comments with enhanced processing"""
    try:
        params = {
            'key': YOUTUBE_API_KEY,
            'videoId': video_info.video_id,
            'part': 'snippet,replies',
            'maxResults': min(max_comments, 100),  # API limit is 100
            'order': 'relevance'
        }

        url = f"{YOUTUBE_API_BASE_URL}/commentThreads"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to load comments: {response.status}")
                    return []

                data = await response.json()

                comments = []
                for item in data.get('items', []):
                    comment_snippet = item['snippet']['topLevelComment']['snippet']

                    comment_info = CommentInfo(
                        comment_id=item['id'],
                        text=comment_snippet['textDisplay'],
                        author=comment_snippet['authorDisplayName'],
                        like_count=comment_snippet.get('likeCount', 0),
                        published_at=comment_snippet['publishedAt'],
                        summary=None,
                        summary_embedding=None,
                        reply_count=item['snippet'].get('totalReplyCount', 0)
                    )
                    comments.append(comment_info)

                logger.info(f"Scraped {len(comments)} comments from video: {video_info.title}")
                comments = comments[:max_comments]

                # Generate summaries and embeddings
                comment_texts = [c.text for c in comments]
                summaries = await analyze_comments_enhanced(comment_texts)

                # Process each comment
                for i, comment in enumerate(comments):
                    if i < len(summaries):
                        comment.summary = summaries[i]
                        # Generate embedding for the summary
                        if comment.summary:
                            comment.summary_embedding = generate_embedding(comment.summary)

                return comments

    except Exception as e:
        logger.error(f"Error scraping comments: {str(e)}")
        return []


async def analyze_comments_enhanced(comments: List[str]) -> List[str]:
    """Enhanced comment analysis with better prompts"""
    try:

        text = ""
        for c in comments:
            text += c + "\n"

        prompt = f"""
        Extract key insights from these YouTube comments. For each comment, identify:
        - Main sentiment (positive/negative/neutral)
        - Core issue or praise mentioned
        - Specific pain points or benefits
        - Actionable business intelligence

        Transform each comment into a concise, structured insight that captures:
        1. The emotional tone
        2. The specific issue/benefit
        3. The business implication

        Comments to analyze:
        {text}

        Return one insight per line, in the same order as the input comments.
        Format: "[SENTIMENT] Customer reports [SPECIFIC_ISSUE/BENEFIT] - [BUSINESS_IMPLICATION]"

        Example: "[NEGATIVE] Customer reports slow delivery times - Logistics improvement needed"
        Don't introduce your data, with a header like "Here are ..."!!!
        """

        chat_completion = groq.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "You are a customer insights analyst. Extract structured, actionable intelligence from customer feedback. Focus on specific issues, emotions, and business implications."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=400
        )

        all_summaries = chat_completion.choices[0].message.content.strip().split('\n')
        all_summaries = [s.strip() for s in all_summaries if s.strip()]

        logger.info(f"Generated {len(all_summaries)} comment summaries")
        return all_summaries

    except Exception as e:
        logger.error(f"Error analyzing comments: {str(e)}")
        return []


async def save_comments_to_supabase(comments: List[CommentInfo], session_id: str, video_info: VideoInfo):
    """Save comments to Supabase table using batch insert"""
    try:
        # Prepare all comment data for batch insertion
        comments_data = []
        for comment in comments:
            comment_data = {
                'id': str(uuid.uuid4()),
                'comment_text': comment.text,
                'session_id': session_id,
                'source': f"YouTube - {video_info.channel}",
                'video_id': video_info.video_id,
                'comment_id': comment.comment_id,
                'summary': comment.summary,
                'summary_embedding': comment.summary_embedding
            }
            comments_data.append(comment_data)

        # Perform batch insert
        result = supabase.table('feedback').insert(comments_data).execute()

        if result.data:
            logger.info(f"Successfully saved {len(result.data)} comments in batch")
        else:
            logger.error("Failed to save comments in batch")

    except Exception as e:
        logger.error(f"Error saving comments to Supabase: {str(e)}")



class Message(Model):
    message: str
    session_id: str


class Response(Model):
    comments_processed: int
    videos_analyzed: int


@agent.on_event("startup")
async def agent_startup(ctx: Context):
    ctx.logger.info("🚀 Enhanced YouTube Manager Agent started")
    ctx.logger.info(f"📍 Agent Address: {agent.address}")

    # Check API keys
    if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        ctx.logger.warning("⚠️ YouTube API Key not set! Please replace YOUTUBE_API_KEY.")

    if SUPABASE_URL == "YOUR_SUPABASE_URL":
        ctx.logger.warning("⚠️ Supabase URL not set! Please replace SUPABASE_URL.")

    # Check embedding model
    if embedding_model is None:
        ctx.logger.warning("⚠️ Embedding model not loaded. Vector embeddings will be disabled.")


@agent.on_rest_post("/register", Message, Response)
async def handle_register(ctx: Context, req: Message) -> Response:
    """Enhanced registration handler with improved processing"""
    ctx.logger.info("Received Register request")
    session_id = req.session_id

    try:
        # Analyze business with enhanced prompt
        business_analysis = await analyze_business_with_llm(req.message)
        logger.info(f"Business analysis: {business_analysis}")

        # Generate search queries with enhanced prompt
        search_queries = generate_search_queries_with_llm(req.message)
        logger.info(f"Generated queries: {search_queries}")

        # Search for videos
        all_videos = []
        for query in search_queries:
            videos = await search_youtube_videos(query, 2)  # Increased per query
            all_videos.extend(videos)

        # Remove duplicates
        unique_videos = []
        seen_ids = set()
        for video in all_videos:
            if video.video_id not in seen_ids:
                unique_videos.append(video)
                seen_ids.add(video.video_id)

        videos_to_process = unique_videos[:12]  # Process more videos
        logger.info(f"Processing {len(videos_to_process)} unique videos")

        # Process comments with enhanced analysis
        total_comments_processed = 0
        for video in videos_to_process:
            comments = await scrape_video_comments(video, 10)  # More comments per video

            if comments:
                # Save to Supabase
                await save_comments_to_supabase(comments, session_id, video)
                total_comments_processed += len(comments)

                # Log sample for debugging
                ctx.logger.info(f"Video: {video.title}")
                ctx.logger.info(f"Sample comments with summaries: {[(c.text[:100], c.summary) for c in comments[:2]]}")

        ctx.logger.info(f"✅ Session {session_id} completed:")
        ctx.logger.info(f"   - Videos analyzed: {len(videos_to_process)}")
        ctx.logger.info(f"   - Comments processed: {total_comments_processed}")
        ctx.logger.info(
            f"   - Comments with embeddings: {sum(1 for video in videos_to_process for comment in [] if comment.summary_embedding)}")

        return Response(
            comments_processed=total_comments_processed,
            videos_analyzed=len(videos_to_process)
        )

    except Exception as e:
        ctx.logger.error(f"Agent error: {str(e)}")
        return Response(
            comments_processed=0,
            videos_analyzed=0
        )


if __name__ == "__main__":
    agent.run()
