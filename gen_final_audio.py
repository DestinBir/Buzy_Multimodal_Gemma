import re
from gtts import gTTS

with open("VIDEO_SCRIPT.md") as f:
    content = f.read()

# Extract all narrator text
parts = []
for m in re.finditer(r'\*\*Narrator:\*\*', content):
    start = m.end()
    end = content.find("\n---", start)
    if end == -1:
        end = len(content)
    chunk = content[start:end].strip()
    # Remove leading > and whitespace from each line
    chunk = re.sub(r'^\s*>\s*', '', chunk, flags=re.MULTILINE)
    chunk = re.sub(r'\s+', ' ', chunk).strip()
    if chunk:
        parts.append(chunk)

full = " ".join(parts)
full = re.sub(r'\*+', '', full)

words = full.split()
print(f"Text: {len(words)} words, est ~{len(words)//2.5:.0f}s")

if len(words) > 450:
    full = " ".join(words[:450])
    print(f"Trimmed to 450 words")

tts = gTTS(text=full, lang="en", tld="com", slow=False)
tts.save("assets/audio/buzy_ai_demo.mp3")

import subprocess
r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1", "assets/audio/buzy_ai_demo.mp3"],
    capture_output=True, text=True)
if r.returncode == 0:
    dur = float(r.stdout.strip())
    print(f"Duration: {dur:.1f}s ({dur/60:.1f} min)")
else:
    print("Saved assets/audio/buzy_ai_demo.mp3")
