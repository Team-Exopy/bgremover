import os
from io import BytesIO

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Try loading rembg
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# U2NET model import
from u2Net.model.u2net import U2NET

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models/u2net/u2net_human_seg.pth")

# Load U2NET model
print("Loading U2NET model from:", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"U2NET model not found at {MODEL_PATH}")

device = torch.device("cpu")
u2net_model = U2NET(3, 1)
state = torch.load(MODEL_PATH, map_location=device)
u2net_model.load_state_dict(state)
u2net_model.eval()
print("U2NET model loaded successfully.")
templates = Jinja2Templates(directory="templates")


# --- Helpers ---
def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)

def _pil_to_tensor(pil_img: Image.Image, size=(320, 320)) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return tfm(pil_img).unsqueeze(0)

def _u2net_prob_mask(image_bytes: bytes) -> np.ndarray:
    pil = Image.open(BytesIO(image_bytes)).convert('RGB')
    W, H = pil.size
    inp = _pil_to_tensor(pil)

    with torch.no_grad():
        d1, *_ = u2net_model(inp)
        mask = d1[0][0].cpu().numpy()
        prob = _safe_normalize(mask)

    prob = cv2.resize((prob * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    return prob.astype(np.float32) / 255.0

def _composite_white(image_bytes: bytes, alpha_mask: np.ndarray) -> BytesIO:
    original = Image.open(BytesIO(image_bytes)).convert('RGBA')

    if (original.size[0], original.size[1]) != (alpha_mask.shape[1], alpha_mask.shape[0]):
        alpha_mask = cv2.resize(alpha_mask, (original.size[0], original.size[1]), interpolation=cv2.INTER_LINEAR)

    alpha_mask = cv2.GaussianBlur(alpha_mask, (3, 3), 0)
    alpha = Image.fromarray(alpha_mask).convert('L')

    subject = original.copy()
    subject.putalpha(alpha)

    arr = np.array(subject)
    alpha_f = arr[..., 3:4] / 255.0
    arr[..., :3] = (arr[..., :3] * alpha_f + 255 * (1 - alpha_f)).astype(np.uint8)

    result = Image.fromarray(arr[..., :3], 'RGB')
    output = BytesIO()
    result.save(output, format='JPEG', quality=95)
    output.seek(0)
    return output

def _make_qr_transparent(image_bytes: bytes, tolerance: int = 30) -> BytesIO:
    """Make black pixels transparent in a QR code"""
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    arr = np.array(image, dtype=np.uint8)
    
    # Extract channels
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    
    # Detect black pixels - all channels must be below threshold
    black_threshold = tolerance
    is_black = (r <= black_threshold) & (g <= black_threshold) & (b <= black_threshold)
    
    # Set black pixels transparent
    arr[is_black, 3] = 0
    
    # Save back to PNG with transparency
    output_img = Image.fromarray(arr, mode="RGBA")
    output_bytes = BytesIO()
    output_img.save(output_bytes, format="PNG", optimize=True)
    output_bytes.seek(0)
    return output_bytes

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/remove-bg")
async def remove_bg(file: UploadFile, is_qr: str = Form(default="false")):
    is_qr = is_qr.lower() == "true"
    content = await file.read()
    filename_base = os.path.splitext(file.filename)[0]

    try:
        if is_qr:
            print("Processing QR Code: making white background transparent...")
            output_bytes = _make_qr_transparent(content, tolerance=30)
            return StreamingResponse(
                output_bytes,
                media_type="image/png",
                headers={"Content-Disposition": f"inline; filename={filename_base}_transparent.png"}
            )
        else:
            print("Processing Photo: removing background...")
            used_rembg = False
            if REMBG_AVAILABLE:
                print("Trying rembg first...")
                rgba = rembg_remove(content)
                cut = Image.open(BytesIO(rgba)).convert('RGBA')
                alpha = np.array(cut.split()[-1])
                if alpha.max() > 0:
                    output_bytes = _composite_white(content, alpha)
                    used_rembg = True

            if not used_rembg:
                print("Falling back to U2NET...")
                prob = _u2net_prob_mask(content)
                alpha = (prob * 255).astype(np.uint8)
                output_bytes = _composite_white(content, alpha)

            return StreamingResponse(
                output_bytes,
                media_type="image/jpeg",
                headers={"Content-Disposition": f"inline; filename={filename_base}_nobg.jpg"}
            )

    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")