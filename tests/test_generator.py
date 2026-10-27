from PIL import Image

from virk.eval.generator import ShiftGenerator


def test_nested_images_with_same_name_are_not_overwritten(tmp_path):
    paths = []
    for camera, value in (("a", 40), ("b", 200)):
        directory = tmp_path / camera
        directory.mkdir()
        image = directory / "frame.png"
        Image.new("RGB", (8, 8), (value, value, value)).save(image)
        paths.append(str(image))
    generated = ShiftGenerator(str(tmp_path / "out")).generate(
        paths, shifts=["brightness"]
    )
    first, second = generated["brightness_s3"]
    assert first != second
    with Image.open(first) as image_a, Image.open(second) as image_b:
        assert image_a.getpixel((0, 0)) != image_b.getpixel((0, 0))
