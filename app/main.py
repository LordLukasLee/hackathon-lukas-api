from dotenv import load_dotenv

load_dotenv()  # Load env vars before other imports that may need them

import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import urllib.parse  # noqa: E402
from typing import Literal  # noqa: E402

import ollama  # noqa: E402
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
        "description": "Retail technology pioneer founded by Andreas Hassellöf. Creator of the Ombori Grid platform that powers digital experiences in physical spaces. Microsoft Azure partner featured by Satya Nadella. Serves major retailers including H&M, Dufry, Target Australia, and Lindt. Recently launched StoreAI for AI-powered in-store experiences.",
        "target_audience": "Retail executives, CIOs, store operations directors at mid-to-large retailers looking to digitally transform their physical stores",
        "voice": "Visionary yet proven. We've delivered results for the world's biggest retailers. Practical innovation, not hype.",
        "content_themes": [
            "Customer experience transformation",
            "ROI of smart store technology",
            "Retail innovation trends",
            "Case studies from H&M, Dufry, etc.",
            "StoreAI and the future of retail",
        ],
        "hashtags_branded": ["Ombori", "OmboriGrid", "StoreAI"],
        "hashtags_industry": [
            "RetailTech",
            "SmartStores",
            "RetailInnovation",
            "DigitalTransformation",
            "CustomerExperience",
        ],
    },
    "phygrid": {
        "name": "Phygrid",
        "description": "The Store Digitalization Standard - edge AI infrastructure deployed by retailers in 50+ countries. Partners with Microsoft Azure and VoiceComm to support 52,000+ retail locations. Complete suite including self-checkout, smart fitting rooms, endless aisle, scan & go, queue management, and digital signage.",
        "target_audience": "Retail IT leaders, operations managers, and innovation teams seeking proven, scalable store digitalization solutions",
        "voice": "The industry standard. Trusted globally. We make complex technology simple to deploy and manage.",
        "content_themes": [
            "Self-checkout best practices",
            "Smart fitting room ROI",
            "Edge AI in retail",
            "Global deployment success stories",
            "Operational efficiency gains",
        ],
        "hashtags_branded": ["Phygrid", "StoreDigitalization"],
        "hashtags_industry": [
            "SelfCheckout",
            "RetailAI",
            "EdgeComputing",
            "SmartRetail",
            "RetailOperations",
        ],
    },
    "phystack": {
        "name": "Phystack",
        "description": "Edge-native infrastructure platform powering the real-world AI revolution. Provides SDK, APIs, and edge intelligence tools for developers building physical-world applications. The foundation that powers Phygrid and enables physical AI across industries.",
        "target_audience": "Developers, CTOs, and technical architects building edge AI and physical-world applications",
        "voice": "Developer-first. We built the infrastructure so you can focus on your application. Technical depth with practical simplicity.",
        "content_themes": [
            "Edge AI development tutorials",
            "Physical AI use cases",
            "Developer tools and SDK updates",
            "Edge vs cloud architecture",
            "Building real-world AI applications",
        ],
        "hashtags_branded": ["Phystack", "PhysicalAI"],
        "hashtags_industry": [
            "EdgeAI",
            "EdgeComputing",
            "DevTools",
            "IoT",
            "AIInfrastructure",
        ],
    },
    "fendops": {
        "name": "Fendops",
        "description": "IT operations and security services for fintech and financial institutions. Part of the Ombori Group. Specializes in transaction security, PCI DSS/FFIEC compliance, and third-party risk management. Helps financial services companies protect sensitive data and maintain regulatory compliance.",
        "target_audience": "CISOs, IT security leaders, compliance officers at fintech companies and financial institutions",
        "voice": "Security expertise you can trust. We understand the regulatory landscape and the real threats facing financial services.",
        "content_themes": [
            "Fintech security best practices",
            "PCI DSS compliance tips",
            "Third-party risk management",
            "Transaction security trends",
            "Regulatory compliance updates",
        ],
        "hashtags_branded": ["Fendops"],
        "hashtags_industry": [
            "Fintech",
            "CyberSecurity",
            "Compliance",
            "PCIDSS",
            "FinancialServices",
            "InfoSec",
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


async def generate_image_with_pollinations(
    prompt: str, style: str = "photo"
) -> str | None:
    """Generate an image using Pollinations API."""
    style_suffix = IMAGE_STYLES.get(style, IMAGE_STYLES["photo"])
    full_prompt = f"{prompt}, {style_suffix}"

    try:
        logger.info(f"Generating image with prompt: {full_prompt[:100]}...")

        # Pollinations API - URL encode the prompt and add API key if available
        encoded_prompt = urllib.parse.quote(full_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

        # Add API key for no rate limits
        api_key = os.getenv("POLLINATIONS_API_KEY")
        if api_key:
            image_url = f"{image_url}?key={api_key}"
            logger.info("Using Pollinations API key for authenticated request")

        logger.info(f"Generated image URL: {image_url[:100]}...")
        return image_url
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None


@app.post("/generate-image", response_model=GenerateImageResponse)
async def generate_image(request: GenerateImageRequest):
    """Generate an image using Pollinations API (free)"""
    style_suffix = IMAGE_STYLES.get(request.style, IMAGE_STYLES["photo"])
    full_prompt = f"{request.prompt}, {style_suffix}"

    image_url = await generate_image_with_pollinations(request.prompt, request.style)
    if not image_url:
        raise HTTPException(
            status_code=500,
            detail="Image generation failed",
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
Target Audience: {target_audience}
Brand Voice: {voice}
Content Themes: {content_themes}

Generate 5 specific, actionable content ideas that would resonate with our target audience. Each idea should be something we could post TODAY.

Content types that perform well for B2B:
- Customer success metrics ("How [client type] achieved X% improvement...")
- Contrarian takes on industry trends
- Behind-the-scenes of technology/process
- Problem → Solution stories from our expertise
- Industry data with our unique perspective

Each idea needs:
- title: Specific, scroll-stopping headline (not generic clickbait)
- description: The angle, key points to cover, and why our audience cares

IMPORTANT: Respond ONLY with valid JSON array. No markdown, no code blocks:
[{{"title": "specific headline", "description": "angle and key points"}}]"""


@app.get("/ideas/{company_id}", response_model=IdeasResponse)
async def get_content_ideas(company_id: str):
    """Generate content ideas for a specific company using AI"""
    if company_id not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company '{company_id}' not found")

    company = COMPANIES[company_id]
    prompt = IDEAS_PROMPT.format(
        company_name=company["name"],
        company_desc=company["description"],
        target_audience=company["target_audience"],
        voice=company["voice"],
        content_themes=", ".join(company["content_themes"]),
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


GENERATE_PROMPT = """You are a social media expert creating content for {company_name}.

Company: {company_desc}
Target Audience: {target_audience}
Brand Voice: {voice}
Topic: "{topic}"
Tone: {tone}

PLATFORM REQUIREMENTS:

INSTAGRAM (visual storytelling, broad reach):
- Start with emoji + bold hook that stops the scroll
- 2-3 short paragraphs with \\n\\n between them
- Share value: insight, tip, or story - not just promotion
- End with engaging question to drive comments
- Hashtags: Use these branded tags {hashtags_branded} plus 4-5 from {hashtags_industry}
- image_suggestion: Describe a visually striking image (not generic stock photo vibes)

LINKEDIN (B2B thought leadership):
- Open with insight, surprising stat, or contrarian take
- Share expertise and real perspective (sound like a human expert, not a brand)
- Can use short bullet points for clarity
- End with question that invites professional discussion
- Hashtags: 3-4 from {hashtags_industry}
- image_suggestion: Professional but not boring - data viz, team photo, or product in action

TWITTER/X (conversation starter):
- Single complete thought - punchy take, question, or insight
- MUST be self-contained (never "Here are 5 things:" without listing them)
- Write to spark replies or retweets
- Under 250 characters including hashtags
- Hashtags: 1-2 only if they add value
- image_suggestion: Eye-catching visual that complements the tweet

Respond with ONLY valid JSON:
{{"instagram":{{"content":"...","hashtags":[...],"image_suggestion":"..."}},"linkedin":{{"content":"...","hashtags":[...],"image_suggestion":"..."}},"twitter":{{"content":"...","hashtags":[...],"image_suggestion":"..."}}}}"""


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
        target_audience=company["target_audience"],
        voice=company["voice"],
        topic=request.topic,
        tone=request.tone,
        hashtags_branded=company["hashtags_branded"],
        hashtags_industry=company["hashtags_industry"],
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
                for platform in ["instagram", "linkedin", "twitter"]:
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
        }

        if request.generate_images:
            logger.info(f"Generating images with style: {request.image_style}")
            for platform in ["instagram", "linkedin", "twitter"]:
                image_suggestion = data[platform].get("image_suggestion", "")
                if image_suggestion:
                    image_url = await generate_image_with_pollinations(
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
        )

    except HTTPException:
        raise
    except ollama.ResponseError as e:
        logger.error(f"Ollama error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating content: {e}")
