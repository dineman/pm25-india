```{=typst}
#show raw.where(block: true): set text(size: 7.6pt)
#set page(numbering: none)
#align(center + horizon)[
  #text(30pt, weight: "bold")[PM2.5 over India]
  #v(0.4em)
  #text(17pt)[Machine learning for spatial enhancement of air-quality maps]
  #v(2.2em)
  #text(13pt, style: "italic")[A beginner's guide to the whole project:\ the data, the 12 steps, every Python file, and what we found]
  #v(4em)
  #text(11pt)[Course project 3 · CAMS EAC4 reanalysis · India · 2020–2024]
  #v(0.6em)
  #text(10pt, fill: rgb("#6b6a63"))[Repository: github.com/dineman/pm25-india · Booklet written October 2026]
]
#pagebreak()
#outline(title: [Contents], depth: 2, indent: auto)
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)
#set heading(numbering: "1.1")
```

# How to read this booklet {-}

This booklet explains a complete data-science project from the very beginning.
You do **not** need to know air pollution science, statistics or machine learning
to follow it: every idea is introduced in plain words the first time it appears,
and the most important terms are collected in the glossary at the end.

The booklet has four parts:

- **Part I, The big picture** (chapters 1–4): what PM2.5 is, where the data comes
  from, what machine learning is, and which questions the project asks.
- **Part II, The project step by step** (chapters 5–17): the twelve steps of the
  work in the order they were done, each with *what* was done, *why*, *how*, and
  *what we found*, followed by the answers in short.
- **Part III, The code** (chapters 18–22): a guided tour of the repository and
  of **every Python file**, with short pieces of code explained line by line.
- **Part IV, Using and judging the project** (chapters 23–26): how to run it
  yourself, the mistakes we found and fixed, the limits of the results, and the
  glossary.

If you are in a hurry, read chapter 4 (the questions) and chapter 17 (the
answers). If you want to change the code, read Part III.

Each step chapter starts with an **"In one sentence"** line that tells you the
main idea before the details.

A note on numbers: PM2.5 is measured in **micrograms per cubic metre of air**,
written **µg m⁻³** (a microgram is a millionth of a gram). India's national
24-hour limit is 60 µg m⁻³; the World Health Organization's guideline is
15 µg m⁻³.
