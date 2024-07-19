from PIL import Image
import matplotlib.pyplot as plt

# Open an image file
image_path = '/user/work/yf20630/cow-dataset-project/images_sample/pmfeed_4_3_16_frame_21937_cow_2.jpg'
img = Image.open(image_path)

# Resize image to 224x224
img_resized = img.resize((224, 224))

# Save the resized image to the root directory
save_path = './resized_image.jpg'
img_resized.save(save_path)

# Display the resized image
plt.imshow(img_resized)
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()

print(f"Resized image saved to {save_path}")
