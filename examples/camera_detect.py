import braid
from PIL import Image, ImageDraw

# Open an image.
print('Image file (default ./truck_sample.jpg):')
filename = input()
img = Image.open('truck_sample.png' if filename == '' else filename)

# Detect axles.
result = braid.axle_groups_from_image(img, site='sentvid')

# If successful, show the results.
if result['success']:
    # Optional: mark the detected vehicle.
    segment = result['segment']
    x = segment['x']
    y = segment['y']
    w = segment['w']
    h = segment['h']

    draw = ImageDraw.Draw(img)
    draw.rectangle([(x, y), (x + w, y + h)], outline='red', width=2)

    print(f"Found {result['vehicle_type']} with probability {result['type_probability']}")
    img.save('detected_vehicle.png')

    # Print detected groups and probabilities.
    for i, (axle_groups, probability) in enumerate(result['axle_groups']):
        print(f"Model {i} recognized axle groups {axle_groups} with probability {probability}.")

# Otherwise, print the error message.
else:
  print(result['msg'])


