
# Example: Using the corrected stippling algorithm

from corrected_stippling import corrected_image_stippling

# Basic usage
points = corrected_image_stippling(
    image_path="your_image.png",
    n_points=3000,
    use_lloyd=True,
    use_blue_noise=True
)

# This will create:
# - corrected_stippling_your_image_3000pts_comparison.png (6-panel comparison)
# - corrected_stippling_your_image_3000pts.txt (point coordinates)
