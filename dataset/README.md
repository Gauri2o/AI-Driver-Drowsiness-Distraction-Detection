@"
# Dataset

This directory contains the local driver-behavior image dataset used for feature extraction.

Subjects:
- S01
- S02

ML training conditions:
- normal
- glasses
- mask

Behavior classes:
- attentive
- drowsy
- yawning

Low-light samples are intentionally excluded from ML training.

Raw dataset images are excluded from the Git repository using `.gitignore`.
"@ | Set-Content -Encoding UTF8 .\dataset\README.md