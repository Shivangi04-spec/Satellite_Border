import os
import cv2
import matplotlib.pyplot as plt
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
CLASSES = ["land", "water", "vegetation", "roads", "buildings"]
print("===== TRAIN DATA =====")
for cls in CLASSES:
    path = os.path.join(TRAIN_DIR, cls)
    if not os.path.exists(path):
        print(f"{cls}: folder NOT FOUND")
    else:
        print(f"{cls}: {len(os.listdir(path))} images")
print("\n===== TEST DATA =====")
for cls in CLASSES:
    path = os.path.join(TEST_DIR, cls)
    if not os.path.exists(path):
        print(f"{cls}: folder NOT FOUND")
    else:
        print(f"{cls}: {len(os.listdir(path))} images")
sample_class = "land"
sample_folder = os.path.join(TRAIN_DIR, sample_class)
sample_image_name = os.listdir(sample_folder)[0]
sample_image_path = os.path.join(sample_folder, sample_image_name)
image = cv2.imread(sample_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("\nSample image shape:", image.shape)
plt.imshow(image)
plt.title("Sample Satellite Image - Land")
plt.axis("off")
plt.show()
