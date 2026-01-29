from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


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


# --- Sample Data Model ---
class Item(BaseModel):
    id: int
    name: str
    description: str
    price: float


# --- Sample Data (replace with database later) ---
SAMPLE_ITEMS = [
    Item(id=1, name="Widget", description="A useful widget for your desk", price=9.99),
    Item(id=2, name="Gadget", description="A fancy gadget with buttons", price=19.99),
    Item(id=3, name="Gizmo", description="An amazing gizmo that does things", price=29.99),
]


@app.get("/")
async def root():
    return {"message": "Hackathon API is running!", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/items", response_model=list[Item])
async def get_items():
    """Get all items - sample data to demonstrate the API pattern"""
    return SAMPLE_ITEMS


@app.get("/items/{item_id}")
async def get_item(item_id: int):
    """Get a specific item by ID"""
    for item in SAMPLE_ITEMS:
        if item.id == item_id:
            return item
    return {"error": "Item not found"}
