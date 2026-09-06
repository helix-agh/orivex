# Documentation development

The site uses MkDocs Material, mkdocstrings with the Python handler, and PyMdown Extensions,
following lonkit's documentation stack. MathJax renders equations. Google-style docstrings
are the convention for new API documentation.

## Preview and build

From the repository root:

```bash
uv sync --locked --extra docs
uv run --no-sync mkdocs serve
```

Open `http://127.0.0.1:8000/orivex/`. The preview reloads when documentation, source code,
the catalogue generator, or branding assets change.

Build the static site with the same validation as CI:

```bash
uv run --no-sync mkdocs build --strict
```

The generated site is written to `site/`, which is ignored by Git. Broken internal links,
missing navigation entries, unresolved API objects, and other build warnings fail the build.
This check does not validate availability of external websites or execute example code.

To work on code and documentation in one environment, install both extras:
`uv sync --extra dev --extra docs`. With pip, use `python -m pip install -e ".[docs]"`.

## Editing content

- Write guides in `docs/` and add new pages to `nav` in `mkdocs.yml`.
- Keep the README focused on installation and a short example; put detailed behavior in guides.
- Edit Python signatures and docstrings in `src/orivex/`; mkdocstrings reads them statically,
  so building the Torch reference does not require installing PyTorch.
- Edit feature metadata in the registered `FeatureSpec` instances. `tools/generate_docs.py`
  rebuilds `features/catalogue.md` during each build; there is no generated Markdown to commit.
- Keep shared logos in `assets/`; the generator includes them in the site without duplicating
  the source images.

Use relative Markdown links between pages so links work under the `/orivex/` Pages prefix.
For equations, use `$...$` inline and `$$...$$` for display math. MathJax is re-run after
Material's instant navigation changes the page.

## GitHub Pages deployment

The [Documentation workflow](https://github.com/helix-agh/orivex/blob/main/.github/workflows/docs.yml)
builds the site on pull requests and pushes to `main`. Pull requests only validate the build.
Pushes to `main` upload the site and deploy it to:

[https://helix-agh.github.io/orivex/](https://helix-agh.github.io/orivex/)

A manual workflow run on `main` also builds and deploys the site. Runs on other branches only
build it. Builds use the documentation extra and the committed `uv.lock`.

For the first deployment, a repository administrator must select **Settings → Pages →
Build and deployment → Source → GitHub Actions**. The deployment job uses the `github-pages`
environment and GitHub's built-in token; no personal token or `gh-pages` branch is required.
Only the deployment job receives `pages: write` and `id-token: write` permissions.

See [GitHub's custom Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
for repository and environment settings.
