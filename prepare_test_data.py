import numpy as np
import imageio.v2 as imageio
import os
import math

def create_digit_image(digit, size=32, thickness=3):
    """Create a simple digit image using only numpy and math"""
    img = np.zeros((size, size), dtype=np.uint8)
    
    patterns = {
        0: [(0.2, 0.2, 0.8, 0.8), (0.3, 0.2, 0.3, 0.8), (0.7, 0.2, 0.7, 0.8)],
        1: [(0.5, 0.2, 0.5, 0.8)],
        2: [(0.2, 0.2, 0.8, 0.2), (0.8, 0.2, 0.8, 0.5), (0.8, 0.5, 0.2, 0.5), (0.2, 0.5, 0.2, 0.8), (0.2, 0.8, 0.8, 0.8)],
        3: [(0.2, 0.2, 0.8, 0.2), (0.8, 0.2, 0.8, 0.8), (0.2, 0.5, 0.8, 0.5), (0.2, 0.8, 0.8, 0.8)],
        4: [(0.2, 0.2, 0.2, 0.5), (0.2, 0.5, 0.8, 0.5), (0.8, 0.2, 0.8, 0.8)],
        5: [(0.2, 0.2, 0.8, 0.2), (0.2, 0.2, 0.2, 0.5), (0.2, 0.5, 0.8, 0.5), (0.8, 0.5, 0.8, 0.8), (0.2, 0.8, 0.8, 0.8)],
        6: [(0.2, 0.2, 0.8, 0.2), (0.2, 0.2, 0.2, 0.8), (0.2, 0.5, 0.8, 0.5), (0.8, 0.5, 0.8, 0.8), (0.2, 0.8, 0.8, 0.8)],
        7: [(0.2, 0.2, 0.8, 0.2), (0.8, 0.2, 0.8, 0.8)],
        8: [(0.2, 0.2, 0.8, 0.2), (0.2, 0.2, 0.2, 0.8), (0.8, 0.2, 0.8, 0.8), (0.2, 0.5, 0.8, 0.5), (0.2, 0.8, 0.8, 0.8)],
        9: [(0.2, 0.2, 0.8, 0.2), (0.2, 0.2, 0.2, 0.5), (0.8, 0.2, 0.8, 0.8), (0.2, 0.5, 0.8, 0.5), (0.2, 0.8, 0.8, 0.8)]
    }
    
    if digit not in patterns:
        return img
    
    for x1, y1, x2, y2 in patterns[digit]:
        px1 = int(x1 * size)
        py1 = int(y1 * size)
        px2 = int(x2 * size)
        py2 = int(y2 * size)
        
        dx = abs(px2 - px1)
        dy = abs(py2 - py1)
        sx = 1 if px1 < px2 else -1
        sy = 1 if py1 < py2 else -1
        err = dx - dy
        
        x, y = px1, py1
        while True:
            for i in range(-thickness//2, thickness//2 + 1):
                for j in range(-thickness//2, thickness//2 + 1):
                    nx, ny = x + i, y + j
                    if 0 <= nx < size and 0 <= ny < size:
                        img[ny, nx] = 255
            
            if x == px2 and y == py2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    return img

def create_realistic_digit(digit, size=32):
    img = create_digit_image(digit, size, thickness=2)
    
    noise = np.random.normal(0, 10, (size, size)).astype(np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    kernel = np.array([[0.1, 0.1, 0.1],
                      [0.1, 0.2, 0.1],
                      [0.1, 0.1, 0.1]])
    
    # Simple convolution
    padded = np.pad(img, 1, mode='edge')
    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(size):
        for j in range(size):
            blurred[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
    
    return np.clip(blurred, 0, 255).astype(np.uint8)

def main():
    test_dir = "test-data"
    os.makedirs(test_dir, exist_ok=True)
    
    print("Creating test digit images...")
    
    for digit in range(10):
        # Create multiple variations
        for variant in range(3):
            img = create_realistic_digit(digit, size=32)
            
            # convert to RGB
            img_rgb = np.stack([img, img, img], axis=-1)
            
            filename = f"digit_{digit}_v{variant}.png"
            filepath = os.path.join(test_dir, filename)
            imageio.imwrite(filepath, img_rgb)
            print(f"Created {filename}")
    
    print(f"Created {30} test images in {test_dir}/")
    print("Images are named: digit_X_vY.png where X is the digit (0-9) and Y is the variant (0-2)")

if __name__ == "__main__":
    main() 