# -- coding: utf-8 --
from PIL import Image
from pathlib import Path

save_path = Path("post")

def make_gif_template():
    gifname = "dcgan_mono_face73_1900_2000.gif"
    img_path = Path("generated_images\\dcgan_mono_face73")
    pattern="face_19*_0000.png"
    fig_number=100
    make_gif_animation_core(gifname, img_path, pattern, fig_number)

def make_gif_animation_core(gifname='out.gif', img_path=Path("generated_images\\dcgan_mono_face73"), pattern="*.png", fig_number=100):
    paths = img_path.glob(pattern)
    dname = save_path / Path(gifname)
    dname.mkdir(parents=True, exist_ok=True)
    fname = dname / Path(gifname)

    imgs = []
    for i, path in enumerate(paths):
        if i < fig_number:
            imgs.append(Image.open(path))
        else:
            break

    imgs[0].save(str(fname), save_all=True, append_images=imgs[1:], duration=250, loop=2)


def main():
    pass

if __name__ == '__main__':
    make_gif_template()