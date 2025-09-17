# CHANGELOG.md

## v1.0.4 (17-09-2025)

**Fixed:**
- Fixed an issue with validating return types for typed dict

**Added:**
- Added a new `vibe_mode` configuration to `VibeConfig` which can be "CHILL", "EAGER" or "AGGRESSIVE" (more information [**here**](https://github.com/vibe-engineers/vibetools/wiki/Tutorial#vibeconfig))

## v1.0.3 (06-09-2025)

**Miscellaneous:**
- Refactored system_instruction default to be provided by sub-libraries

## v1.0.2 (05-09-2025)

**Fixed:**
- Removed hard dependency on openai and google-genai

## v1.0.1 (05-09-2025)

**Fixed:**
- Added missing check for eval input type

**Miscellaneous:**
- Moved exceptions and config exports into submodules

## v1.0.0 (04-09-2025)

**Added:**
- Initial Release