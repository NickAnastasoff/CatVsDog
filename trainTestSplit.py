import os

def move_images(source_dir, cat_dir, dog_dir):
  """Moves all images from the source directory into the cat and dog directories."""
  for img, filename in enumerate(os.listdir(source_dir)):
    if img % 5 == 0:
        if filename.startswith("cat"):
            os.rename(os.path.join(source_dir, filename), os.path.join(cat_dir, filename))
        elif filename.startswith("dog"):
            os.rename(os.path.join(source_dir, filename), os.path.join(dog_dir, filename))
        print("Moved {} to {}".format(filename, cat_dir if filename.startswith("cat") else dog_dir))
 
if __name__ == "__main__":
  source_dir = "train/dogs"
  cat_dir = "test/cats"
  dog_dir = "test/dogs"
  move_images(source_dir, cat_dir, dog_dir)
