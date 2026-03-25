from PIL import Image

from capx.integrations.vision.molmo import init_molmo

image_path = "scripts/images/table.jpg"
image = Image.open(image_path)

molmo_det_fn = init_molmo()
points = molmo_det_fn(
    image, objects=["handle of the square nut", "square nut center", "square block"]
)
print(points)
