from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_FOLDER = './private_samples/'

def compute_mad(ela_image):
    ela_array = np.array(ela_image)
    return np.mean(ela_array)

def compute_stddev(ela_image):
    ela_array = np.array(ela_image)
    return np.std(ela_array)

def compute_high_intensity_ratio(ela_image, threshold=50):
    ela_array = np.array(ela_image)
    high_intensity_pixels = np.sum(ela_array > threshold)
    total_pixels = ela_array.size
    return (high_intensity_pixels / total_pixels) * 100

def error_level_analysis(image_path):
    original = Image.open(image_path).convert('RGB')
    original.save("temp.jpg", 'JPEG', quality = 75)
    recompressed = Image.open("temp.jpg")

    diff = ImageChops.difference(original, recompressed)
    diff = ImageEnhance.Brightness(diff).enhance(5)
    print(compute_mad(diff))
    print(compute_stddev(diff))
    print(compute_high_intensity_ratio(diff))

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(original)
    
    plt.subplot(1,2,2)
    plt.title("Error Level Analysis")
    plt.imshow(diff)
    
    plt.show()

def perform_ela(image_path, quality=90, scale=15):
    original = Image.open(image_path).convert('RGB')
    temp_path = "temp.jpg"
    original.save(temp_path, 'JPEG', quality = quality)

    recompressed = Image.open(temp_path).convert('RGB')
    diff = ImageChops.difference(original, recompressed)

    extrema = diff.getextrema()
    max_diff = max([ ex[1] for ex in extrema]) or 1
    scale_factor = 255.0 / max_diff if max_diff > 0 else scale
    enhanced = ImageEnhance.Brightness(diff).enhance(scale_factor)

    print("Maximum difference was %d" % (max_diff))

    return enhanced

ela_image = perform_ela(SAMPLE_FOLDER + "selfie_1.jpg")
ela_image.show()