# profile_photos

Face-centered 800×800 headshots for the About Us page. Every person on
`aboutus/index.html` is produced by the same pipeline so that all cards
look consistent (face centered, roughly the same apparent size).

## Layout

```
profile_photos/
├── face_crop.py   # the cropping script
├── original/      # canonical source photo, one per person
├── adjusted/      # optional AI- or manually-preprocessed source
│                  #   (upscale, outpaint, unblur, enhance) — only the
│                  #   people who needed it have a file here
└── cropped/       # final 800×800 output; this is what the HTML loads
```

Filename convention: `{firstname-lastname}.{ext}` in `original/`,
`{firstname-lastname}-adjusted.{ext}` in `adjusted/`,
`{firstname-lastname}-cropped.png` in `cropped/`.

## Pipeline

1. Drop the canonical source in `original/{name}.{ext}`.
2. If the source is low-resolution, blurry, or doesn't have enough room
   around the face for a co-centered crop, produce a preprocessed
   variant and save it as `adjusted/{name}-adjusted.{ext}`. Typical
   tools: Claude-AI-assisted outpainting, [evoto.ai](https://evoto.ai)
   face enhance/unblur, a non-generative upscaler.
3. Run `face_crop.py` on the best available source (adjusted if present,
   otherwise original) to produce `cropped/{name}-cropped.png`.
4. Reference `../images/profile_photos/cropped/{name}-cropped.png` from
   `aboutus/index.html`.

## Running the script

```
# process one photo
python face_crop.py original/jane-doe.jpg cropped/jane-doe-cropped.png

# visualize the detected face, fitted ellipse, eye line, and crop circle
python face_crop.py cropped/jane-doe-cropped.png /tmp/debug.png --debug
```

Dependencies: `pip install insightface onnxruntime opencv-python`.
First run downloads the InsightFace `buffalo_l` model (~300 MB) to
`~/.insightface/models/`.

## Calibration

The crop ratios are pinned to Scott Selikoff's reference photo
(`original/scott-selikoff.png`, 600×600, face ellipse diameter 222):

- `CROP_TO_FACE_DIAM = 2.70` — crop side / face ellipse diameter
- `TARGET_SIZE = 800` — output side length in pixels
- Face is centered on the fitted-ellipse center of the 106 facial
  landmarks, so the face sits in the middle of the circular mask on the
  website.

If you ever swap out the detector or the reference photo, re-derive
these constants from the new baseline.

## Padding behavior

When the ideal crop window extends past the edge of the source image,
`extract_crop` fills the overshoot with `cv2.BORDER_REPLICATE` (the
outermost pixel row/column is smeared outward). This is good enough for
small overruns (a few dozen pixels). For larger overruns — roughly, when
the script prints pad values over ~40 px on any side — the smear starts
to look stretched, and the source should be outpainted into
`adjusted/{name}-adjusted.{ext}` instead.
