# Q-DRIVE Alpha - Git Setup Guide

## Quick Start

```bash
# 1. Initialize git repository (if not already done)
cd ~/qdrive_alpha
git init

# 2. Check for large files BEFORE adding anything
./check_large_files.sh

# 3. Add files
git add .

# 4. Check what will be committed
git status

# 5. Create first commit
git commit -m "Initial commit: Core modules and utilities"

# 6. Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/qdrive_alpha.git

# 7. Push to GitHub
git push -u origin main
```

---

## What Gets Tracked (.gitignore Strategy)

### ✅ INCLUDED (Will be committed)

**Directories:**
- `/Core/` - All core system modules
- `/temp/` - Temporary development files
- `/custom_assets/` - Custom assets you've created
- `/images/` - Project images
- `/jetson/` - Jetson-specific code
- `/Tools/` - Development tools
- `/Utility/` - Utility scripts
- `/configs/` - Configuration files

**Main Directory Scripts:**
- `Main.py`
- `PreWelcomeSelect.py`
- `DataConsolidator.py`
- `DataIngestion.py`
- `ScenarioManager.py`
- `ScenarioLibrary.py`
- `PredictiveManager.py`
- `PredictiveIndices.py`
- `EventManager.py`
- `DynamicMonitor.py`
- `Helpers.py`
- `HUD.py`
- `input_utils.py`
- `monitor_config.py`
- `controls_queue_s.py`
- `dynamic_mapping.py`
- `automated_migration_script.py`
- `dir_setup.sh`

**Project Files:**
- `README.md`
- `.gitignore`
- `requirements.txt`
- `Dockerfile`
- Any other `.md` files

### ❌ EXCLUDED (Will NOT be committed)

**Large Directories:**
- `CarlaUE4/` - CARLA game engine (gigabytes)
- `Engine/` - Unreal Engine files
- `HDMaps/` - Large map files
- `LLM/` - Language model files
- `audio/`, `TTS/` - Audio files
- `Backups/` - Old backups
- `Session_logs/` - Runtime logs
- `PythonAPI/` - CARLA Python API
- `scenario_runner*/` - External tools
- `Research/` - Research materials
- `Documentation/` - Documentation (if you want to track, remove from .gitignore)

**File Types (Anywhere):**
- `*.mp4`, `*.avi`, `*.mov` - Videos
- `*.mp3`, `*.wav` - Audio
- `*.pth`, `*.pt`, `*.h5`, `*.ckpt` - ML models
- `*.zip`, `*.tar.gz`, `*.rar` - Archives
- `*.pkl`, `*.pickle`, `*.npy` - Data files
- `*.log` - Log files
- `__pycache__/`, `*.pyc` - Python cache

**Patterns:**
- Any folder named `data/`, `datasets/`, `logs/`
- Any file with `backup` or `old` in the name
- IDE files (`.vscode/`, `.idea/`)

---

## GitHub File Size Limits

| Size | Status | Action |
|------|--------|--------|
| < 50 MB | ✅ OK | Files commit normally |
| 50-100 MB | ⚠️ Warning | GitHub warns but allows |
| > 100 MB | ❌ Blocked | **Cannot push to GitHub** |

### If You Have Files >100MB

**Option 1: Use Git LFS (Large File Storage)**
```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.pth"
git lfs track "*.onnx"
git lfs track "*.mp4"

# Commit the .gitattributes file
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

**Option 2: Move to External Storage**
- Store large files on Google Drive, Dropbox, S3
- Reference them in README or a `download_models.sh` script

**Option 3: Exclude from Git**
- Add to `.gitignore`
- Document where team should get these files

---

## Useful Commands

### Check What Will Be Committed
```bash
# See all files that will be tracked
git status

# See files that are being ignored
git status --ignored

# List all tracked files
git ls-files

# See file sizes of tracked files
git ls-files | xargs du -h | sort -h
```

### Before First Push - Verify
```bash
# 1. Run the large file checker
./check_large_files.sh

# 2. Do a dry-run check
git add .
git status

# 3. Check total repository size
du -sh .git

# 4. If all looks good, commit
git commit -m "Initial commit"

# 5. Push
git push -u origin main
```

### If You Accidentally Committed Large Files

```bash
# Remove file from git but keep locally
git rm --cached path/to/large/file

# Add to .gitignore
echo "path/to/large/file" >> .gitignore

# Commit the change
git add .gitignore
git commit -m "Remove large file from tracking"

# If already pushed, you need to rewrite history (dangerous!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/large/file" \
  --prune-empty --tag-name-filter cat -- --all
```

### Clean Up Git History (Remove all large files)
```bash
# Use BFG Repo-Cleaner (safer than filter-branch)
# Download from: https://rtyley.github.io/bfg-repo-cleaner/

# Remove files larger than 100MB
java -jar bfg.jar --strip-blobs-bigger-than 100M .git

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

## Customizing .gitignore

### Track a Specific File in an Ignored Directory
```bash
# In .gitignore, add:
!HDMaps/my_important_map.xodr
```

### Ignore a Specific File in a Tracked Directory
```bash
# In .gitignore, add:
Core/Vision/experimental_test.py
```

### Temporarily Ignore Changes to a Tracked File
```bash
# Don't track changes, but don't remove from git
git update-index --assume-unchanged path/to/file

# To start tracking again
git update-index --no-assume-unchanged path/to/file
```

---

## Best Practices

### ✅ DO:
- Run `./check_large_files.sh` before first commit
- Keep repository under 1GB total if possible
- Use meaningful commit messages
- Commit often with small, logical changes
- Keep sensitive data (API keys, passwords) out of git
- Review `git status` before each commit

### ❌ DON'T:
- Commit large binary files (models, videos, datasets)
- Commit generated files (logs, cache, __pycache__)
- Commit IDE-specific files (.vscode/, .idea/)
- Commit credentials or API keys
- Force push to main/master branch
- Commit without checking what's staged

---

## Setting Up GitHub Repository

### Creating a New Repository on GitHub
1. Go to https://github.com/new
2. Repository name: `qdrive_alpha` (or your choice)
3. **Keep it PRIVATE** (recommended for development)
4. **Don't** initialize with README (you already have one)
5. Click "Create repository"

### Connecting Local to GitHub
```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/qdrive_alpha.git

# Verify
git remote -v

# Push (first time)
git push -u origin main

# Push (subsequent)
git push
```

### Using SSH Instead of HTTPS (Recommended)
```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings > SSH and GPG keys > New SSH key

# Update remote to use SSH
git remote set-url origin git@github.com:YOUR_USERNAME/qdrive_alpha.git
```

---

## Troubleshooting

### "Repository Too Large" Error
```bash
# Check repository size
du -sh .git

# Find large objects
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort --numeric-sort --key=2 | \
  tail -20

# Clean up
git gc --aggressive --prune=now
```

### "File Exceeds 100MB" Error
```bash
# Find the large file
git lfs migrate info

# Either:
# 1. Add to .gitignore and remove from git
# 2. Use Git LFS
# 3. Split into smaller files
```

### Accidentally Committed Sensitive Data
```bash
# Remove from history IMMEDIATELY
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive/file" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (if already pushed)
git push --force

# Change any exposed credentials ASAP!
```

---

## Files Created for You

1. **`.gitignore`** - Main ignore file (already configured)
2. **`check_large_files.sh`** - Utility to find large files before committing
3. **`GIT_SETUP_GUIDE.md`** - This guide

---

## Quick Reference Commands

```bash
# Status check
git status                    # What will be committed
git status --ignored          # What is being ignored
./check_large_files.sh       # Find large files

# Commit workflow
git add .                     # Stage all changes
git commit -m "message"       # Commit with message
git push                      # Push to remote

# Viewing history
git log                       # Commit history
git log --oneline            # Compact history
git diff                     # See unstaged changes
git diff --staged            # See staged changes

# Undoing things
git restore <file>           # Discard unstaged changes
git restore --staged <file>  # Unstage file
git reset --soft HEAD~1      # Undo last commit (keep changes)
git reset --hard HEAD~1      # Undo last commit (discard changes)

# Branches
git branch                   # List branches
git checkout -b new-branch   # Create and switch to branch
git merge branch-name        # Merge branch into current
```

---

## Need Help?

- GitHub Docs: https://docs.github.com
- Git Cheat Sheet: https://education.github.com/git-cheat-sheet-education.pdf
- Git LFS: https://git-lfs.github.com
- BFG Repo-Cleaner: https://rtyley.github.io/bfg-repo-cleaner/

---

**Generated for Q-DRIVE Alpha Project**
**Date:** 2025-10-14
