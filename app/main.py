from dotenv import load_dotenv

load_dotenv()  # Load env vars before other imports that may need them

import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from typing import Literal  # noqa: E402

import ollama  # noqa: E402
import replicate  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Image Style Presets ---
IMAGE_STYLES = {
    "photo": "professional photography, high quality, natural lighting, photorealistic",
    "illustration": "digital illustration, vector art style, vibrant colors, clean lines",
    "infographic": "infographic style, data visualization, clean design, minimal text",
    "minimalist": "minimalist design, clean white background, simple shapes, modern aesthetic",
    "3d": "3D render, octane render, studio lighting, photorealistic, cinematic",
}


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling various formats."""
    # Try to find JSON in code blocks first
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object with regex
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to find JSON array with regex
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not extract valid JSON from response")


app = FastAPI(
    title="Hackathon API",
    description="Backend API for Ombori Hackathon",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Company Profiles ---
COMPANIES = {
    "ombori": {
        "name": "Ombori",
        "description": "Retail technology pioneer founded by Andreas Hassellöf. Creates smart store solutions including IoT, digital signage, queue management, and interactive displays. Partners with Microsoft and serves major retailers like H&M.",
        "topics": [
            "retail innovation",
            "customer experience",
            "smart stores",
            "digital transformation",
            "IoT",
        ],
    },
    "phygrid": {
        "name": "Phygrid",
        "description": "The Store Digitalization Standard - edge AI infrastructure for retail deployed in 50+ countries. Offers self-checkout, smart fitting rooms, scan & go, endless aisle, and digital signage solutions.",
        "topics": [
            "self-checkout",
            "smart fitting rooms",
            "retail AI",
            "store digitalization",
            "edge computing",
        ],
    },
    "phystack": {
        "name": "Phystack",
        "description": "Edge AI infrastructure platform powering physical-world applications. One platform for everything physical in digital spaces - from screens to drones, managing devices, apps, and real-world analytics globally.",
        "topics": [
            "edge AI",
            "physical AI",
            "IoT infrastructure",
            "real-world analytics",
            "developer platform",
        ],
    },
    "fendops": {
        "name": "Fendops",
        "description": "Technology integration and enablement services helping organizations with Microsoft Teams, Slack, Zoom, ERP, CRM, and HR systems. Specializes in change management and business process integration.",
        "topics": [
            "digital workplace",
            "enterprise integration",
            "change management",
            "collaboration tools",
            "business transformation",
        ],
    },
}


@app.get("/")
async def root():
    return {"message": "Hackathon API is running!", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# --- Image Generation ---


class ImageStyle(BaseModel):
    id: str
    name: str
    description: str


@app.get("/image-styles", response_model=list[ImageStyle])
async def get_image_styles():
    """Get list of available image styles"""
    style_names = {
        "photo": "Photo",
        "illustration": "Illustration",
        "infographic": "Infographic",
        "minimalist": "Minimalist",
        "3d": "3D Render",
    }
    return [
        ImageStyle(id=style_id, name=style_names[style_id], description=desc)
        for style_id, desc in IMAGE_STYLES.items()
    ]


class GenerateImageRequest(BaseModel):
    prompt: str
    style: str = "photo"


class GenerateImageResponse(BaseModel):
    image_url: str
    prompt_used: str


async def generate_image_with_replicate(
    prompt: str, style: str = "photo"
) -> str | None:
    """Generate an image using Replicate's Flux Schnell (free tier friendly)."""
    if not os.environ.get("REPLICATE_API_TOKEN"):
        logger.warning("REPLICATE_API_TOKEN not set, skipping image generation")
        return None

    style_suffix = IMAGE_STYLES.get(style, IMAGE_STYLES["photo"])
    full_prompt = f"{prompt}, {style_suffix}"

    try:
        logger.info(f"Generating image with prompt: {full_prompt[:100]}...")
        # Use Flux Schnell - fast and free tier friendly
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": full_prompt,
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 90,
                "go_fast": True,
            },
        )
        # Output is a list of FileOutput objects
        if output and len(output) > 0:
            image_url = str(output[0])
            logger.info(f"Generated image: {image_url}")
            return image_url
        return None
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None


@app.post("/generate-image", response_model=GenerateImageResponse)
async def generate_image(request: GenerateImageRequest):
    """Generate an image using Stable Diffusion via Replicate"""
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise HTTPException(
            status_code=503,
            detail="REPLICATE_API_TOKEN not configured. Set it in your environment.",
        )

    style_suffix = IMAGE_STYLES.get(request.style, IMAGE_STYLES["photo"])
    full_prompt = f"{request.prompt}, {style_suffix}"

    image_url = await generate_image_with_replicate(request.prompt, request.style)
    if not image_url:
        raise HTTPException(
            status_code=402,
            detail="Image generation failed. Please add billing credit at https://replicate.com/account/billing",
        )

    return GenerateImageResponse(image_url=image_url, prompt_used=full_prompt)


# --- Company Endpoints ---


class Company(BaseModel):
    id: str
    name: str
    description: str


@app.get("/companies", response_model=list[Company])
async def get_companies():
    """Get list of available companies"""
    return [
        Company(id=cid, name=c["name"], description=c["description"])
        for cid, c in COMPANIES.items()
    ]


# --- Content Ideas ---


class ContentIdea(BaseModel):
    title: str
    description: str


class IdeasResponse(BaseModel):
    company: str
    ideas: list[ContentIdea]


IDEAS_PROMPT = """You are a senior social media strategist for {company_name}.

Company: {company_desc}

Generate 5 specific, actionable content ideas. Each idea should be something they could post TODAY - not vague concepts but concrete topics with a clear angle.

Good examples:
- "How we reduced checkout times by 40% at [major retailer]" (specific case study)
- "3 things I wish I knew before implementing self-checkout" (listicle with insider tips)
- "The hidden cost of long checkout lines (and how to fix it)" (problem-solution)

Bad examples (too vague):
- "The future of retail" (too broad)
- "Innovation in technology" (no angle)

For each idea, provide:
- title: A specific, scroll-stopping headline (not generic)
- description: What the post will cover and why it's interesting

IMPORTANT: Respond ONLY with valid JSON array. No markdown, no code blocks:
[{{"title": "specific headline", "description": "what this post covers and the angle"}}]"""


@app.get("/ideas/{company_id}", response_model=IdeasResponse)
async def get_content_ideas(company_id: str):
    """Generate content ideas for a specific company using AI"""
    if company_id not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company '{company_id}' not found")

    company = COMPANIES[company_id]
    prompt = IDEAS_PROMPT.format(
        company_name=company["name"], company_desc=company["description"]
    )

    try:
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.8},
        )

        response_text = response["message"]["content"]
        ideas_data = extract_json(response_text)

        ideas = [
            ContentIdea(title=i["title"], description=i["description"])
            for i in ideas_data
        ]

        return IdeasResponse(company=company["name"], ideas=ideas)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating ideas: {e}")


# --- Social Media Content Generator ---


class GenerateRequest(BaseModel):
    company_id: str
    topic: str
    tone: Literal["professional", "casual", "fun"] = "professional"
    generate_images: bool = False
    image_style: str = "photo"


class PlatformContent(BaseModel):
    content: str
    hashtags: list[str]
    char_count: int
    image_suggestion: str = ""  # Description of ideal visual for this post
    image_url: str | None = None  # URL of generated image
    image_style: str = "photo"  # Style used for image generation


class GeneratedContent(BaseModel):
    company: str
    topic: str
    instagram: PlatformContent
    linkedin: PlatformContent
    twitter: PlatformContent
    tiktok: PlatformContent


GENERATE_PROMPT = """You are a JSON API. You MUST respond with ONLY a JSON object, nothing else. No explanations, no markdown.

Task: Create social media posts for {company_name} about "{topic}" in {tone} tone.
Company info: {company_desc}

IMPORTANT: Use \\n\\n between paragraphs in content strings for visual spacing.

Requirements per platform:
- instagram: 150-250 words, emoji hook, 2-3 paragraphs separated by \\n\\n, question at end, 6-8 hashtags
- linkedin: 150-250 words, professional insight, 2-3 paragraphs separated by \\n\\n, question at end, 3-4 hashtags
- twitter: under 250 chars total, punchy, 1-2 hashtags
- tiktok: 30-60 words, POV: or hook format, \\n\\n after hook, 4-5 hashtags with #fyp

Each platform needs: content (string), hashtags (array), image_suggestion (string describing ideal photo/visual)

YOUR RESPONSE MUST BE EXACTLY THIS FORMAT (valid JSON only, no other text):
{{"instagram":{{"content":"emoji hook\\n\\nparagraph1\\n\\nparagraph2\\n\\nquestion?","hashtags":["tag1","tag2"],"image_suggestion":"describe ideal photo"}},"linkedin":{{"content":"hook\\n\\nparagraph1\\n\\nparagraph2\\n\\nquestion?","hashtags":["tag1"],"image_suggestion":"describe ideal graphic"}},"twitter":{{"content":"your tweet","hashtags":["tag1"],"image_suggestion":"describe visual"}},"tiktok":{{"content":"POV: hook\\n\\ncaption","hashtags":["fyp","tag1"],"image_suggestion":"describe video idea"}}}}"""


@app.post("/generate", response_model=GeneratedContent)
async def generate_content(request: GenerateRequest):
    """Generate social media content for 4 platforms using local Ollama"""
    if request.company_id not in COMPANIES:
        raise HTTPException(
            status_code=404, detail=f"Company '{request.company_id}' not found"
        )

    company = COMPANIES[request.company_id]
    prompt = GENERATE_PROMPT.format(
        company_name=company["name"],
        company_desc=company["description"],
        topic=request.topic,
        tone=request.tone,
    )

    try:
        # Try up to 3 times in case of JSON parsing issues
        last_error = None
        response_text = ""
        data = None
        for attempt in range(3):
            response = ollama.chat(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.7 if attempt == 0 else 0.3,
                    "num_predict": 8000,  # Ensure enough tokens for full response
                },
            )
            response_text = response["message"]["content"]
            logger.info(
                f"Attempt {attempt + 1} raw response length: {len(response_text)}"
            )

            try:
                data = extract_json(response_text)
                # Validate required keys exist
                for platform in ["instagram", "linkedin", "twitter", "tiktok"]:
                    if platform not in data:
                        raise ValueError(f"Missing platform: {platform}")
                    if "content" not in data[platform]:
                        raise ValueError(f"Missing content for {platform}")
                    if "hashtags" not in data[platform]:
                        data[platform]["hashtags"] = []  # Default to empty
                break
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                logger.warning(f"Raw response: {response_text[:500]}")
                last_error = e
                continue

        if data is None:
            logger.error(f"All attempts failed. Last response: {response_text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse LLM response after 3 attempts: {last_error}",
            )

        def build_platform(
            platform_data: dict,
            image_url: str | None = None,
            image_style: str = "photo",
        ) -> PlatformContent:
            content = platform_data.get("content", "")
            hashtags = platform_data.get("hashtags", [])
            image_suggestion = platform_data.get("image_suggestion", "")
            clean_hashtags = [h.lstrip("#") for h in hashtags if isinstance(h, str)]
            hashtag_text = " ".join(f"#{h}" for h in clean_hashtags)
            full_text = f"{content}\n\n{hashtag_text}" if clean_hashtags else content
            return PlatformContent(
                content=full_text,
                hashtags=clean_hashtags,
                char_count=len(full_text),
                image_suggestion=image_suggestion,
                image_url=image_url,
                image_style=image_style,
            )

        # Generate images if requested
        image_urls: dict[str, str | None] = {
            "instagram": None,
            "linkedin": None,
            "twitter": None,
            "tiktok": None,
        }

        if request.generate_images:
            logger.info(f"Generating images with style: {request.image_style}")
            for platform in ["instagram", "linkedin", "twitter", "tiktok"]:
                image_suggestion = data[platform].get("image_suggestion", "")
                if image_suggestion:
                    image_url = await generate_image_with_replicate(
                        image_suggestion, request.image_style
                    )
                    image_urls[platform] = image_url

        return GeneratedContent(
            company=company["name"],
            topic=request.topic,
            instagram=build_platform(
                data["instagram"], image_urls["instagram"], request.image_style
            ),
            linkedin=build_platform(
                data["linkedin"], image_urls["linkedin"], request.image_style
            ),
            twitter=build_platform(
                data["twitter"], image_urls["twitter"], request.image_style
            ),
            tiktok=build_platform(
                data["tiktok"], image_urls["tiktok"], request.image_style
            ),
        )

    except HTTPException:
        raise
    except ollama.ResponseError as e:
        logger.error(f"Ollama error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating content: {e}")
