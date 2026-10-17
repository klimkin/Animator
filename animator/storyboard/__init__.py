"""The --prompt pipeline, independent of which LLM or narrator runs it.

Read order (see pipeline.py):
    STEP 1  script.py    idea -> scenes of narrated beats
    STEP 2  images.py    scene -> whiteboard image
    STEP 3  detect.py    image + beats -> region plan
    STEP 4  narrator     beats -> one take per scene + per-beat windows (../providers/),
                         alongside steps 2-3
    STEP 5  pipeline.py  render every scene into one video

schemas.py holds the structured-output models. Prompt text is in ../prompts/,
backends are in ../providers/.
"""
