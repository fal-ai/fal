"""Image and video results shown inline in the terminal.

preview.py is the glue the `fal dev` commands call and imports no Pillow;
term_media.py (images, protocols, layout) and term_video.py (ffmpeg frames)
load Pillow and are imported only once something is shown.
"""
