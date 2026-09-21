# Round 4 - Social Engine Revival

**Competition:** Data Vortex | AARUUSH'26  
**Theme:** Rebuilding the Social Engine  
**Team:** Entropy  
**Members:** Saanvi Grover & Aditya Sharma

---

## What this is

An interactive **Streamlit + Plotly** dashboard that integrates all four rounds into one narrative: data recovery, SQL analytics, language understanding and live social monitoring.

**The design rule:** nothing is hard-coded. Every figure on every tab is computed at load time from that round's own artefacts - the cleaned CSVs, the SQLite database, the saved model files and the collected live dataset. If a notebook is re-run, the dashboard changes with it.

## Run it

```bash
pip install -r Entropy_requirements.txt
streamlit run Entropy_app.py
```

Opens at `http://localhost:8501`. The dark theme is pinned in `.streamlit/config.toml` so the dashboard renders identically regardless of the viewer's OS theme. It is laid out for a desktop window; below roughly 900 px wide Streamlit stacks the columns and the chart grid becomes a single column.

## Layout

There is no sidebar and there are no tabs. **The four-round pipeline itself is the navigation:** a flowchart of five connected nodes sits permanently below the brand bar, stays pinned to the top of the window as you scroll, and every node is a button. Each carries its round label, its stage name and its headline number, and the active node is lit in that round's accent colour, so the dashboard always shows where you are in the pipeline and what came before it.

```
Round 1 P1     ->  Round 1 P2       ->  Round 2            ->  Round 3          ->  Round 4
Data Recovery      Analytical Core      Semantic Recovery      Signal Tracking      The Revived Engine
```

Round 4 is the last node **and the landing page**, because Round 4 is not a separate analysis - it is this dashboard. You arrive at the finished engine and walk backwards through how it was built. There is deliberately no separate "overview" page: an overview of the four rounds and the synthesis of them are the same document, so they are one page.

Below the rail, **every page opens with a strip of KPI tiles**, then a grid of charts, then the round's method as a vertical flowchart beside its artefact panel. Interpretation text lives inside an `Interpretation` expander beneath each figure, so the visuals lead and the reasoning is one click away. Only the selected page is computed, so switching rounds is instant.

| Page | What it covers | Live computation behind it |
|---|---|---|
| **Round 1 P1 - Data Recovery** | The repair ledger, platform volume, engagement distributions, the correlation check, spread by coefficient of variation, the midnight-timestamp artefact, followers vs engagement, and three hypothesis tests | Recomputes everything from the CSVs, filtered live by the **platform multi-select** at the top of the page |
| **Round 1 P2 - Analytical Core** | The constrained schema and the three submitted questions (E3, M4, H4) in **nested tabs**, each with its result chart, result table and the SQL itself | **Executes the three `.sql` files against `Entropy_social_engine.db` in read-only mode** on every page load |
| **Round 2 - Semantic Recovery** | Every candidate ranked by validation and test macro-F1, the confusion matrix, per-class precision/recall/F1, the substring-rule discovery, and a live "read a post" demo | Loads the saved ensemble (TF-IDF + embeddings + 3 MiniLM seeds) and scores text you type |
| **Round 3 - Signal Tracking** | Sentiment mix over time, the four-platform breakdown, activity with detected spikes, theme ranking, the two inference fixes measured against hand labels, every spike matched to a named news event, a **robustness block** (author concentration, a VADER baseline, period intervals and an alert replay), a live **operating-point slider**, and a filterable **evidence table** of the scored corpus | Re-bins the 1,403 scored live posts inside the **time-window slider** and reads the Round 3 metrics file |
| **Round 4 - The Revived Engine** | Headline KPIs spanning all four rounds, the live reaction timeline with the launch marked, the sentiment verdict, one signature chart per round, **the journey end to end in four cross-round charts**, the same diagnostics applied to both corpora, five recommendations, and what we would not claim | Reads every round's artefacts at once and re-runs the cross-corpus comparisons |

## Design choices

- **A background that sits behind, not on top.** The deep-green data artwork is served from `static/Entropy_bg.webp` (Streamlit's `enableStaticServing`, so it is fetched once and cached by the browser rather than base64'd into every rerun) and fixed in place under a scrim that deepens down the page - lightest at the top where the image has room, near-solid lower down where the dense charts are. Cards stay fully opaque, so nothing is ever read against texture.
- **One card shape, everywhere.** Every chart, method step and artefact list sits in the same panel: same radius, same border, same background. A column of charts reads as a grid instead of as loose figures on a page.
- **Four charts that need all four rounds to exist.** The journey section on Round 4 is the part that no single round could produce: what each round actually handled (three separate corpora, not one funnel); every model built with a dotted line showing what the jump to live data cost; a Lorenz curve of engagement in the synthetic corpus against the live one (Gini 0.27 vs 0.85, where the live top 1% carries roughly half of all engagement); and the training label balance (33/33/33) against the world the model actually met (40/41/19).
- **Charts first.** Roughly thirty-four Plotly figures across five pages, in multi-column grids, with hover detail on every mark. A page should read as a dashboard at a glance, not as a document with pictures.
- **Every page ends with its method.** After the charts and the analysis, a vertical flowchart walks the round step by step: what was done, why it was done that way, and the artefact each step produced, with the hand-off to the next round marked as the outcome. Beside it, a sticky artefact panel shows what that round consumed and what it produced, with file sizes read from disk on the page load, so a missing artefact says so rather than being quietly omitted. Six steps for Round 1 Phase 1, six for Phase 2, seven for Round 2, eight for Round 3 and five for Round 4. It is the argument for the charts that follow.
- **Every page ends with its reports and its notebook.** A report picker renders that round's submitted PDFs page by page inside the dashboard - rasterised server-side with `pypdfium2`, so they read the same in any browser, with a page-range slider for the long ones and the original file one click away. Round 4 collects all seven reports in one picker.
- **Every page ends with its own notebook.** The executed `.ipynb` for that round is parsed at load time and rendered in place: the code cells, the outputs they produced (stream text, results and matplotlib figures) and the notebook's own markdown notes. The whole notebook is shown by default inside a bounded scrolling area, so it never buries the rest of the page. Each code cell is a card with an `In [n]` chip and its output beneath a divider, while markdown cells stay as prose against a rule, so narrative and code are told apart at a glance; a section picker keyed to the notebook's own headings jumps to one part when that is all you want, and the raw `.ipynb` is one download away. Round 4 has no notebook of its own, so it carries all seven reports in one picker instead.
- **The engine is usable, not just described.** The landing page carries a live predictor: type a post, or paste a link to a public one, and it is scored by the real saved ensemble using the same chunking and neutral-bias corrections Round 3 applies to every live post. Links are resolved through each platform's own key-less endpoint (Hacker News Algolia, Reddit Atom, Mastodon and Lemmy status APIs) with a plain-page fallback; private and loopback addresses are refused.
- **SHANNON, an assistant that only knows this project.** *Retrieval-Augmented Intelligence by Entropy* - named for the author of information theory, which is also where the team name comes from. A dock at the bottom left answers questions from the repository itself - the four READMEs, every report's LaTeX source, the metrics and robustness JSON and the cleaning log - retrieved with a hybrid word/character TF-IDF index and summarised by a Groq-hosted model, with the source files listed under every answer. It needs `GROQ_API_KEY` in a `.env` beside `Entropy_app.py` (see `.env.example`); without one it says so rather than failing. The model is resolved against what the key can actually reach rather than hard-coded, since Groq's catalogue varies by account.
- **Typography and identity.** One type scale across the whole app: Outfit for the wordmark, section headings, chart titles, KPI values and node titles; Inter at 1.62 line-height for body copy; widget labels and cell gutters as small uppercase tracked text. Section headings carry a gradient rule. Both faces are pinned from Google Fonts so the dashboard looks the same on any machine. The brand bar carries an animated gradient wordmark, an SVG signal mark with a rotating ring, and a colour sweep along the rule beneath it - all of it disabled under `prefers-reduced-motion`.
- **One palette, used consistently.** A deep forest canvas with sage, teal, olive, sand, bronze and clay accents - a categorical set that runs green to clay, so a legend reads as an ordered natural ramp rather than six unrelated hues. Sentiment is always the same three colours - clay Negative, grey-green Neutral, green Positive - on every chart in the app.
- **Every chart is paired with two notes:** *What this shows* (reading the figure) and *Why it matters* (the decision it informs). A chart without an interpretation is decoration.
- **The operating point is a control, not an assertion.** Round 3 shipped a 1.05 neutral bias; the saved per-post probabilities mean any other weight can be applied instantly, with no model in the loop. A slider re-decides all 1,403 posts, rescores the 150 hand labels, and redraws the sentiment timeline live, so a reader can check that the decline holds across the plausible range rather than taking our chosen number on trust. A weight of 1.00 reproduces the submission exactly.
- **The claims are checked against themselves.** The Round 3 page carries the output of `Entropy_03_robustness_checks.py`: whether each period's level is distinguishable from zero at all, whether the negativity is a handful of accounts (it is not - 562 negative posts from 521 authors), whether the reused ensemble beats an off-the-shelf lexicon (0.699 against VADER's 0.496), and when a simple monitoring rule would have fired (19 h after release, no false alarms before it). Both team members labelled the 150 validation posts independently and agree at **Cohen's kappa 0.896**; the 140 they agree on are the reference standard for every accuracy figure on that page.
- **Uncertainty is drawn, not just tested.** The net-sentiment timeline carries a 95% interval from 1,200 multinomial resamples of each bin's class counts, so thin bins look thin and the detected shift has to clear its own error bars.
- **The evidence is one filter away.** The scored corpus is browsable on the Round 3 page - by sentiment, platform, mentioned release, free text and a confidence ceiling - with the three class probabilities beside every post and a link to the original. Lowering the ceiling surfaces what the engine was least sure about, which is where its errors concentrate.
- **Interactive where interaction changes the answer:** the platform filter, the time-window slider, the nested query tabs and the model demo all re-run the underlying computation rather than hiding pre-rendered output.
- **Honest framing.** Truncated axes are called out in the caption, null results are reported as results, and the caveats are on the page (the midnight timestamp artefact, the platform-mix confound, the wide validation interval, the single annotator).
- **Graceful degradation.** If an artefact is missing the page shows which file to regenerate instead of crashing.

## Dependencies on the other rounds

```
Round4-Social-Engine-Revival/Entropy_app.py
  ├── ../Round1-Phase1-Data-Recovery/   cleaned posts + users CSVs, corrupted CSV, cleaning log
  ├── ../Round1-Phase2-Analytical-Core/ Entropy_social_engine.db + queries/*.sql
  ├── ../Round2-Semantic-Recovery/      model/ (ensemble), outputs/ (metrics, predictions), Entropy_nlp_utils.py
  └── ../Round3-Signal-Tracking/        outputs/ (scored posts, metrics), data/ (manifest, news timeline)
```

The app must therefore be run from inside the repository, with the sibling round folders present.

## File structure

```
Round4-Social-Engine-Revival/
|-- Entropy_app.py                 # the dashboard (single file, 5 pages)
|-- Entropy_post_fetcher.py        # resolves a public post URL to its text
|-- Entropy_rag.py                 # the retrieval index and Groq call behind the assistant
|-- .env.example                   # copy to .env and add GROQ_API_KEY
|-- static/Entropy_bg.webp         # background artwork, served at app/static/
|-- Entropy_requirements.txt       # pinned minimum versions
|-- .streamlit/config.toml         # pinned deep-forest theme + static serving
|-- Entropy_README_Round4.md       # this file
```

## Notes

- `st.pdf` is not used: it needs the `streamlit-pdf` component, and version 2.0.1 fails to register against Streamlit 1.56. Rasterising with `pypdfium2` avoids the dependency conflict and does not rely on the browser having a PDF viewer at all.
- Notebook outputs longer than 3,000 characters are clipped, and the clip says how much was removed rather than trailing off silently.
- HTML outputs in the notebooks are skipped by the viewer. They carry Jupyter's own light-theme table styling, which fights the dark page; the plain-text form of the same result is shown instead.
- The Language Model demo loads three fine-tuned transformers, so the **first** prediction takes about a minute on CPU; it is cached afterwards. Every other page renders in under a second.
- A scikit-learn compatibility shim is applied when loading the saved model: the pickles were written by a newer version that no longer sets `multi_class` on `LogisticRegression`, so the attribute is restored to the trained value rather than retraining.
- The repository link sits in the brand bar, top right, under the team name.
- The pipeline diagram on the Overview is drawn with annotated markers rather than a Sankey: Plotly's Sankey link ribbons animate in, and the transition does not always complete when the figure is first laid out inside a Streamlit container.
- The navigation rail is pinned with `position: sticky` applied to the wrapper *around* the rail container, not the container itself - Streamlit sizes a container to its content, so a sticky rail has no room to travel inside its own box.

## Tools

Streamlit, Plotly, pandas, numpy, scipy, scikit-learn, sqlite3, transformers and sentence-transformers.
