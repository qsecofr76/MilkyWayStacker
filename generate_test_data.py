import os
import cv2
import numpy as np

def generate_mock_images():
    output_dir = "test_images"
    os.makedirs(output_dir, exist_ok=True)
    
    width, height = 800, 600
    
    # Define a static set of stars
    np.random.seed(42)
    num_stars = 200
    star_x = np.random.randint(0, width, num_stars)
    star_y = np.random.randint(0, height // 2, num_stars)
    star_brightness = np.random.randint(150, 255, num_stars)

    # 5 frames
    for f in range(5):
        # Create dark background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw sky gradient (dark blue to pitch black)
        for y in range(height // 2 + 100):
            val = int(20 * (1 - y / (height // 2 + 100)))
            img[y, :] = [val + 10, val, 5]

        # Draw stars with slight movement (rotation or translation)
        # Shift stars slightly to the right to simulate Earth's rotation (5 pixels per frame)
        shift_x = f * 8
        for i in range(num_stars):
            x = int(star_x[i] + shift_x) % width
            y = star_y[i]
            # Draw star with subpixel smoothness
            cv2.circle(img, (x, y), 2, (int(star_brightness[i]), int(star_brightness[i]), 255), -1)

        # Draw landscape (bottom part of image)
        # Landscape moves slightly (e.g. tracking error or tracker drift of 2 pixels per frame)
        # to simulate tracked landscape rotation/movement
        drift_x = f * 2
        
        # Ground background (dark gray)
        img[height//2 + 50:, :] = [15, 10, 10]
        
        # Draw some mountains
        pts = np.array([
            [0, height],
            [100 + drift_x, height - 100],
            [250 + drift_x, height - 200],
            [400 + drift_x, height - 80],
            [600 + drift_x, height - 180],
            [700 + drift_x, height - 110],
            [width, height]
        ], np.int32)
        cv2.fillPoly(img, [pts], (25, 30, 25))
        
        # Add some noise to simulate sensor noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        filename = os.path.join(output_dir, f"frame_{f:02d}.jpg")
        cv2.imwrite(filename, img)
        print(f"Generated {filename}")

if __name__ == "__main__":
    generate_mock_images()
