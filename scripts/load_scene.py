import habitat_sim
from habitat_sim.utils.data import ImageExtractor

# Path to your Replica scene file (e.g., apartment_0.glb)
SCENE_FILE = "path/to/replica/apartment_0/habitat/mesh_semantic.ply"

# Initialize extractor
extractor = ImageExtractor(
    SCENE_FILE,
    img_size=(512, 512),
    output=["rgba", "depth"]
)

# Extract a single sample
sample = extractor[0]
rgb = sample["rgba"]   # RGBA image (H, W, 4)
depth = sample["depth"] # Depth map (H, W)