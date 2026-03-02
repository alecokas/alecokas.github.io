# alecokas.github.io

Personal website and machine learning blog (Jekyll).

## Run locally (to test changes)

- **Install Ruby gems**:

```bash
bundle install
```

- **Start the dev server**:

```bash
bundle exec jekyll serve --livereload --open-url
```

- **Open the site**:
  - Jekyll usually serves at `http://127.0.0.1:4000`

Useful extra commands:

```bash
# Build once (like CI / GitHub Pages build step)
bundle exec jekyll build

# Helpful diagnostics
bundle exec jekyll doctor
```

## Fresh laptop setup (macOS)

### System dependencies

- **Xcode Command Line Tools** (needed for native gems like `ffi`, etc.):

```bash
xcode-select --install
```

- **Homebrew**:
  - Install Homebrew normally, then prefer installing packages under **Apple Silicon (ARM64)** on Apple Silicon Macs.
  - If you see errors like “Cannot install under Rosetta 2 in ARM default prefix (/opt/homebrew)”, your terminal is likely running under Rosetta. Use an ARM64 terminal (or run brew commands with `arch -arm64 ...`).

### Ruby + Bundler

- **Ruby version**: Use the Ruby version specified in `Gemfile` (currently `3.1.7`).
- **Ruby manager**: any is fine; pick one you like. Example with `rbenv`:

```bash
brew install rbenv ruby-build
rbenv init

# install and select the repo's Ruby version
rbenv install 3.1.7
rbenv local 3.1.7

# bundler
gem install bundler
bundle install
```

If you use a different Ruby manager (e.g. `mise`, `asdf`, `chruby`), the key is simply: **Ruby matches `Gemfile` + `bundle install` succeeds**.

### Where to put cover images

Pages/posts can opt into cover images via front matter keys like:

- `cover_image`: path to the image (recommended location: `assets/images/...`)
- `cover_image_alt`: alt text
- `cover_image_caption`: optional caption

Example:

```yaml
---
layout: page
title: Research
cover_image: /assets/images/research-cover.jpg
cover_image_alt: Abstract network visualization
cover_image_caption: Recent work and publications
---
```