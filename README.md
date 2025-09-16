<p align="center">
  <img width=300 src="https://raw.githubusercontent.com/vibe-engineers/vibetools/main/assets/vibetools.png" />
  <h1 align="center">VibeTools</h1>
</p>

<p align="center">
  <a href="https://github.com/vibe-engineers/vibetools/actions/workflows/ci-cd-pipeline.yml"> <img src="https://github.com/vibe-engineers/vibetools/actions/workflows/ci-cd-pipeline.yml/badge.svg" /> </a>
  <a href="https://pypi.org/project/vibetools/"><img src="https://img.shields.io/pypi/v/vibetools.svg" /></a>
  <a href="https://pypi.org/project/vibetools/"><img src="https://img.shields.io/pypi/pyversions/vibetools.svg" /></a>
  <a href="https://github.com/vibe-engineers/vibetools/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/vibetools.svg" /></a>
  <a href="https://pepy.tech/project/vibetools"><img src="https://pepy.tech/badge/vibetools" /></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

## Table of Contents
* [Introduction](#introduction)
* [Features](#features)
* [Technologies](#technologies)
* [Team](#team)
* [Contributing](#contributing)
* [Others](#others)

### Introduction
**VibeTools** is a lightweight python package containing shared modules/logic across several **vibe-engineers** libraries. Note that this package is depended on by other sublibraries (e.g. VibeChecks) and is **not intended to be used externally**. Unless you're a developer working on one of the vibe-engineers libraries, you should not be installing this core library directly.

For internal developers, **VibeTools** is published on [**pypi**](https://pypi.org/project/vibetools/) and can be easily installed with:
```bash
python3 -m pip install vibetools
```
Details on the usage of the package and available APIs can be found on the [**wiki page**](https://github.com/vibe-engineers/vibetools/wiki).

### Features
- **Shared LLM Modules**: The core library provides shared LLM modules such as Google Gemini and OpenAI Wrappers. It also contains shared evaluation logic.
- **Common Base Exceptions**: Several common base exceptions are available out of the box (e.g. VibeTimeoutException, VibeResponseParseException).
- **Logger**: A custom logger is included in the core library for ease and consistency of log outputs.

### Technologies
Technologies used by VibeTools are as below:
##### Done with:

<p align="center">
  <img height="150" width="150" src="https://logos-download.com/wp-content/uploads/2016/10/Python_logo_icon.png"/>
</p>
<p align="center">
Python
</p>

##### Project Repository
```
https://github.com/vibe-engineers/vibetools
```

### Team
* [Kong Le-Yi](https://github.com/konglyyy)
* [Tan Jin](https://github.com/tjtanjin)

### Contributing
If you are looking to contribute to the project, you may find the [**Developer Guide**](https://github.com/vibe-engineers/vibetools/blob/main/docs/DeveloperGuide.md) useful.

In general, the forking workflow is encouraged and you may open a pull request with clear descriptions on the changes and what they are intended to do (enhancement, bug fixes etc). Alternatively, you may simply raise bugs or suggestions by opening an [**issue**](https://github.com/vibe-engineers/vibetools/issues) or raising it up on [**discord**](https://discord.gg/dBW35GBCPZ).

Note: Templates have been created for pull requests and issues to guide you in the process.

### Others
For any questions regarding the implementation of the project, you may also reach out on [**discord**](https://discord.gg/dBW35GBCPZ).

