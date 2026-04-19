from tensorflow.keras.preprocessing.image import ImageDataGenerator
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    rescale=1.0/255
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
print("\nClass indices:")
print(train_generator.class_indices)
print("\nTraining samples:", train_generator.samples)
print("Testing samples:", test_generator.samples)
